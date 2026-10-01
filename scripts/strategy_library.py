import numpy as np
import copy
import os
from PIL import Image
import torch
from ldspo_config import COMMISSION_RATE, INITIAL_BALANCE, WINDOW_SIZE, LOOKBACK_WINDOW, STRATEGY_ACTIONS, NUM_EXPERTS, POSITION_CONFIG, ADAPTIVE_STRATEGY_CONFIG
BUY_HOLD_CONFIG = {'position_size': 1.0}

class AdaptiveStrategyExecutor:

    def __init__(self, env):
        self.env = env
        self._prices_cache = {}
        self._indicators_cache = {}
        self.debug_mode = False
        self._isolated_env_cached = None
        self._isolated_executor_cached = None
        self.df = getattr(env, 'df', None)
        self.close_col = getattr(env, 'close_col', 'close')
        self.high_col = getattr(env, 'high_col', 'high')
        self.low_col = getattr(env, 'low_col', 'low')
        self.volume_col = getattr(env, 'volume_col', 'vol')
        self._close_prices_arr = None
        self._high_prices_arr = None
        self._low_prices_arr = None
        self._hist_close_arr = None
        self._hist_high_arr = None
        self._hist_low_arr = None
        self._hist_volume_arr = None
        self._cached_full_indicators = None
        self._cached_atr = {}
        self._build_price_cache()
        self._calculate_market_features()

    def _build_price_cache(self):
        if self.df is not None and len(self.df) > 0:
            if self.close_col in self.df.columns:
                self._close_prices_arr = self.df[self.close_col].to_numpy(dtype=np.float64, copy=False)
            if self.high_col in self.df.columns:
                self._high_prices_arr = self.df[self.high_col].to_numpy(dtype=np.float64, copy=False)
            if self.low_col in self.df.columns:
                self._low_prices_arr = self.df[self.low_col].to_numpy(dtype=np.float64, copy=False)
        if self._close_prices_arr is None and hasattr(self.env, 'prices') and (self.env.prices is not None):
            self._close_prices_arr = np.asarray(self.env.prices, dtype=np.float64)
        if self._high_prices_arr is None and self._close_prices_arr is not None:
            self._high_prices_arr = self._close_prices_arr * 1.01
        if self._low_prices_arr is None and self._close_prices_arr is not None:
            self._low_prices_arr = self._close_prices_arr * 0.99
        try:
            hist_data = self.env.data['analysis_100days']['analysis_data']
            self._hist_close_arr = np.asarray(hist_data.get('CLOSING_PRICES', []), dtype=np.float64)
            self._hist_high_arr = np.asarray(hist_data.get('HIGH_PRICES', []), dtype=np.float64)
            self._hist_low_arr = np.asarray(hist_data.get('LOW_PRICES', []), dtype=np.float64)
            self._hist_volume_arr = np.asarray(hist_data.get('VOLUMES', []), dtype=np.float64)
        except Exception:
            self._hist_close_arr = np.asarray([], dtype=np.float64)
            self._hist_high_arr = np.asarray([], dtype=np.float64)
            self._hist_low_arr = np.asarray([], dtype=np.float64)
            self._hist_volume_arr = np.asarray([], dtype=np.float64)
        self._precompute_indicators_cache()

    def _precompute_indicators_cache(self):
        close = self._close_prices_arr
        high = self._high_prices_arr
        low = self._low_prices_arr
        if close is None or len(close) == 0:
            self._cached_full_indicators = None
            self._cached_atr = {}
            return
        if high is None:
            high = close * 1.01
        if low is None:
            low = close * 0.99
        if self.df is not None and self.volume_col in self.df.columns and (len(self.df) == len(close)):
            volumes = self.df[self.volume_col].to_numpy(dtype=np.float64, copy=False)
        elif len(self._hist_volume_arr) > 0:
            volumes = np.full(len(close), float(np.mean(self._hist_volume_arr)), dtype=np.float64)
        else:
            volumes = np.full(len(close), 1000000.0, dtype=np.float64)
        self._cached_full_indicators = self._recompute_indicators(close, high, low, volumes)
        for period in (14, 20):
            self._cached_atr[period] = self._compute_atr_vectorized(high, low, close, period)

    def _compute_atr_vectorized(self, high, low, close, period=14):
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        n = len(close)
        if n == 0:
            return []
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0] if n > 0 else 0.0
        if n > 1:
            hl = high[1:] - low[1:]
            hc = np.abs(high[1:] - close[:-1])
            lc = np.abs(low[1:] - close[:-1])
            tr[1:] = np.maximum(hl, np.maximum(hc, lc))
        cs_padded = np.empty(n + 1, dtype=np.float64)
        cs_padded[0] = 0.0
        np.cumsum(tr, out=cs_padded[1:])
        indices = np.arange(n)
        starts = np.maximum(0, indices - period + 1)
        window_sums = cs_padded[indices + 1] - cs_padded[starts]
        window_lens = indices - starts + 1
        atr = window_sums / window_lens
        return atr.tolist()

    def _should_use_fallback_strategy(self, start_step, end_step, strategy_name):
        window_size = end_step - start_step
        if window_size < 10:
            return (True, f'window_too_short_{window_size}')
        try:
            if self._close_prices_arr is not None and len(self._close_prices_arr) > 0:
                max_step = len(self._close_prices_arr)
                prices_arr = self._close_prices_arr[start_step:min(end_step, max_step)]
                if len(prices_arr) < 5:
                    return (True, 'insufficient_prices')
                nan_count = int(np.sum(~np.isfinite(prices_arr) | (prices_arr <= 0)))
                if nan_count / len(prices_arr) > 0.3:
                    return (True, f'too_many_nans_{nan_count}/{len(prices_arr)}')
            else:
                max_step = len(self.df) if self.df is not None else len(self.env.prices) if hasattr(self.env, 'prices') and self.env.prices else 0
                prices = [self._price_at(s) for s in range(start_step, min(end_step, max_step))]
                if len(prices) < 5:
                    return (True, 'insufficient_prices')
                nan_count = sum((1 for p in prices if not np.isfinite(p) or p <= 0))
                if nan_count / len(prices) > 0.3:
                    return (True, f'too_many_nans_{nan_count}/{len(prices)}')
        except Exception as e:
            return (True, f'price_read_error_{str(e)[:20]}')
        return (False, '')

    def _execute_fallback_strategy(self, start_step, end_step, strategy_name):
        try:
            lookback = min(10, start_step)
            if lookback > 0:
                hist_start_price = self._price_at(start_step - lookback)
                hist_end_price = self._price_at(start_step)
                ret = hist_end_price / hist_start_price - 1 if hist_start_price > 0 else 0
            else:
                ret = 0
        except Exception:
            ret = 0
        threshold = 0.002
        fallback_reason = f'fallback_{strategy_name}_hist_ret{ret:.3f}'
        if ret > threshold:
            result = self._execute_long_only(start_step, end_step)
            result['strategy_name'] = fallback_reason + '_long'
        elif ret < -threshold:
            if getattr(self.env, 'allow_short', True):
                result = self._execute_short_only(start_step, end_step)
                result['strategy_name'] = fallback_reason + '_short'
            else:
                result = self._execute_cash_only(start_step, end_step)
                result['strategy_name'] = fallback_reason + '_cash_no_short'
        else:
            result = self._execute_cash_only(start_step, end_step)
            result['strategy_name'] = fallback_reason + '_cash'
        return result

    def _get_adaptive_period(self, window_size, default_period, min_period=3, max_ratio=0.4):
        max_period = int(window_size * max_ratio)
        period = min(default_period, max_period)
        period = max(period, min_period)
        return period

    def _price_at(self, step, col_type='close'):
        if col_type == 'high':
            arr = self._high_prices_arr
            col_name = self.high_col
        elif col_type == 'low':
            arr = self._low_prices_arr
            col_name = self.low_col
        else:
            arr = self._close_prices_arr
            col_name = self.close_col
        if arr is not None and 0 <= step < len(arr):
            price = float(arr[step])
        else:
            raise ValueError(f'Cannot read {col_name} at step {step}: price cache unavailable')
        if not np.isfinite(price):
            print(f"\n{'=' * 80}")
            print(f'[ERROR] Non-finite price')
            print(f"{'=' * 80}")
            print(f'  Step: {step}')
            print(f'  Column: {col_name}')
            print(f'  Price: {price}')
            if self.df is not None and step < len(self.df):
                print(f'  Row {step} data:')
                for col in ['close', 'high', 'low', 'open', 'vol']:
                    if col in self.df.columns:
                        print(f'    {col}: {self.df.loc[step, col]}')
            print(f"{'=' * 80}\n")
            raise ValueError(f'Price at step {step} is not finite: {price}')
        if price <= 0:
            print(f"\n{'=' * 80}")
            print(f'[ERROR] Non-positive price')
            print(f"{'=' * 80}")
            print(f'  Step: {step}')
            print(f'  Column: {col_name}')
            print(f'  Price: {price}')
            if self.df is not None and step < len(self.df):
                print(f'  Row {step} data:')
                for col in ['close', 'high', 'low', 'open', 'vol']:
                    if col in self.df.columns:
                        print(f'    {col}: {self.df.loc[step, col]}')
            print(f"{'=' * 80}\n")
            raise ValueError(f'Price at step {step} is non-positive: {price}')
        return price

    def _calculate_market_features(self):
        hist_data = self.env.data['analysis_100days']['analysis_data']
        prices = np.array(hist_data['CLOSING_PRICES'])
        returns = np.diff(prices) / prices[:-1]
        self.volatility = np.std(returns) * np.sqrt(252)
        atr_values = self._calculate_atr()
        if len(atr_values) > 0 and prices[-1] > 0:
            self.atr_ratio = atr_values[-1] / prices[-1]
        else:
            self.atr_ratio = 0.02
        if len(prices) >= 20:
            self.price_range_20d = (max(prices[-20:]) - min(prices[-20:])) / prices[-1]
        else:
            self.price_range_20d = 0.05
        volumes = np.array(hist_data['VOLUMES'])
        if len(volumes) >= 20:
            self.volume_volatility = np.std(volumes[-20:]) / (np.mean(volumes[-20:]) + 1e-06)
        else:
            self.volume_volatility = 0.5

    def _get_adaptive_parameters(self, strategy_name):
        params = {}
        config_params = ADAPTIVE_STRATEGY_CONFIG.get(strategy_name.lower().replace('cross', '_cross'), {})
        if strategy_name == 'MACross':
            if self.volatility > 0.35:
                params['fast_period'] = config_params.get('high_vol_fast', 3)
                params['slow_period'] = config_params.get('high_vol_slow', 12)
            elif self.volatility < 0.18:
                params['fast_period'] = config_params.get('low_vol_fast', 8)
                params['slow_period'] = config_params.get('low_vol_slow', 25)
            else:
                params['fast_period'] = config_params.get('normal_fast', 5)
                params['slow_period'] = config_params.get('normal_slow', 18)
            params['atr_multiplier'] = max(1.0, min(2.0, 1.2 * (1 + self.volatility * 0.3)))
        elif strategy_name == 'Momentum':
            params['lookback'] = config_params.get('lookback', 8)
            base_threshold = config_params.get('entry_threshold_base', 0.15)
            params['entry_threshold'] = max(config_params.get('min_entry', 0.02), min(config_params.get('max_entry', 0.06), self.volatility * base_threshold))
            params['exit_threshold'] = params['entry_threshold'] * config_params.get('exit_threshold_ratio', 0.4)
            params['atr_multiplier'] = max(1.0, min(2.0, 1.2 * (1 + self.volatility * 0.3)))
        elif strategy_name == 'Turtle':
            if self.price_range_20d > config_params.get('range_threshold', 0.08):
                params['entry_window'] = config_params.get('high_range_entry', 12)
                params['exit_window'] = config_params.get('high_range_exit', 8)
            else:
                params['entry_window'] = config_params.get('low_range_entry', 20)
                params['exit_window'] = config_params.get('low_range_exit', 10)
            params['atr_multiplier'] = max(1.2, min(2.5, 1.5 * (1 + self.volatility * 0.3)))
            params['pyramid_atr'] = 0.5
        elif strategy_name == 'Volume':
            params['price_window'] = 20
            params['volume_window'] = 5
            vol_config = config_params
            params['volume_multiplier'] = max(vol_config.get('volume_mult_min', 1.3), min(vol_config.get('volume_mult_max', 2.5), vol_config.get('volume_mult_base', 1.8) * (1 + self.volume_volatility * 0.2)))
            params['ma_period'] = 10
            params['atr_multiplier'] = 1.8
        elif strategy_name == 'ATR':
            params['atr_period'] = 20
            params['ma_period'] = 20
            atr_config = config_params
            params['atr_mult_entry'] = max(atr_config.get('entry_mult_min', 1.2), min(atr_config.get('entry_mult_max', 2.5), atr_config.get('entry_mult_base', 1.5) * (1 + self.volatility * 0.2)))
            params['atr_mult_exit'] = params['atr_mult_entry'] * atr_config.get('exit_ratio', 0.5)
            params['atr_mult_stop'] = params['atr_mult_entry'] * 0.8
        elif strategy_name == 'Boll':
            params['window'] = 20
            boll_config = config_params
            params['num_std'] = max(boll_config.get('std_min', 1.3), min(boll_config.get('std_max', 2.2), boll_config.get('std_base', 1.8) * (1 + self.volatility * 0.1)))
            params['atr_multiplier'] = 1.8
        elif strategy_name == 'RSI':
            rsi_config = config_params
            if self.volatility > 0.3:
                params['oversold'] = rsi_config.get('oversold_high_vol', 30)
                params['overbought'] = rsi_config.get('overbought_high_vol', 70)
            else:
                params['oversold'] = rsi_config.get('oversold_normal', 35)
                params['overbought'] = rsi_config.get('overbought_normal', 65)
            params['neutral'] = 50
            params['atr_multiplier'] = 1.8
        elif strategy_name == 'KDJ':
            kdj_config = config_params
            vol_adjustment = min(kdj_config.get('vol_adjustment', 8), self.volatility * 10)
            params['period'] = 9
            params['j_buy_threshold'] = kdj_config.get('j_buy_base', 15) - vol_adjustment
            params['k_buy_threshold'] = kdj_config.get('k_buy_base', 25) - vol_adjustment
            params['j_sell_threshold'] = kdj_config.get('j_sell_base', 85) + vol_adjustment
            params['k_sell_threshold'] = kdj_config.get('k_sell_base', 75) + vol_adjustment
            params['atr_multiplier'] = 1.8
        else:
            params = {}
        return params

    def _calculate_dynamic_position_size(self, direction='long'):
        base_position = POSITION_CONFIG['base_position']
        vol_thresholds = POSITION_CONFIG['volatility_thresholds']
        position_adjustments = POSITION_CONFIG['position_adjustments']
        if self.volatility < vol_thresholds['low']:
            volatility_adjustment = position_adjustments['low_vol']
        elif self.volatility < vol_thresholds['medium']:
            volatility_adjustment = position_adjustments['normal_vol']
        elif self.volatility < vol_thresholds['high']:
            volatility_adjustment = position_adjustments['medium_vol']
        else:
            volatility_adjustment = position_adjustments['high_vol']
        hist_data = self.env.data['analysis_100days']['analysis_data']
        prices = np.array(hist_data['CLOSING_PRICES'])
        if len(prices) >= 20:
            trend_return = (prices[-1] - prices[-20]) / prices[-20]
            if direction == 'long':
                if trend_return > 0.08:
                    trend_adjustment = 0.05
                elif trend_return > 0.03:
                    trend_adjustment = 0.02
                elif trend_return < -0.03:
                    trend_adjustment = -0.03
                else:
                    trend_adjustment = 0.0
            elif trend_return < -0.08:
                trend_adjustment = 0.05
            elif trend_return < -0.03:
                trend_adjustment = 0.02
            elif trend_return > 0.03:
                trend_adjustment = -0.03
            else:
                trend_adjustment = 0.0
        else:
            trend_adjustment = 0.0
        if 'RSI_VALUES' in hist_data and len(hist_data['RSI_VALUES']) > 0:
            rsi = hist_data['RSI_VALUES'][-1]
            if rsi < 25 or rsi > 75:
                rsi_adjustment = -0.02
            else:
                rsi_adjustment = 0.0
        else:
            rsi_adjustment = 0.0
        position_size = base_position + volatility_adjustment + trend_adjustment + rsi_adjustment
        min_position = POSITION_CONFIG['min_position']
        max_position = POSITION_CONFIG['max_position']
        position_size = max(min_position, min(max_position, position_size))
        return position_size

    def _create_isolated_env(self, initial_capital):
        isolated_env = copy.copy(self.env)
        isolated_env.balance = initial_capital
        isolated_env.initial_balance = initial_capital
        isolated_env.shares_held = 0
        isolated_env.short_shares = 0
        isolated_env.short_entry_price = 0
        isolated_env.portfolio_value = initial_capital
        isolated_env.trades = []
        isolated_env.daily_equity_curve = []
        isolated_env.commission_rate = COMMISSION_RATE
        isolated_env.df = self.env.df
        isolated_env.close_col = self.env.close_col
        isolated_env.high_col = self.env.high_col
        isolated_env.low_col = self.env.low_col
        isolated_env.volume_col = self.env.volume_col
        return isolated_env

    def _reset_isolated_env_state(self, isolated_env, initial_capital):
        isolated_env.balance = initial_capital
        isolated_env.initial_balance = initial_capital
        isolated_env.shares_held = 0
        isolated_env.short_shares = 0
        isolated_env.short_entry_price = 0
        isolated_env.portfolio_value = initial_capital
        isolated_env.trades = []
        isolated_env.daily_equity_curve = []
        isolated_env.current_step = 0
        isolated_env.current_price = isolated_env.prices[0] if hasattr(isolated_env, 'prices') and len(isolated_env.prices) > 0 else 0

    def execute_strategy_isolated(self, strategy_id, start_step, end_step, initial_capital):
        if self._isolated_executor_cached is None or self._isolated_env_cached is None:
            isolated_env = self._create_isolated_env(initial_capital)
            isolated_executor = AdaptiveStrategyExecutor(isolated_env)
            isolated_executor.df = self.df
            isolated_executor.close_col = self.close_col
            isolated_executor.high_col = self.high_col
            isolated_executor.low_col = self.low_col
            isolated_executor.volume_col = self.volume_col
            isolated_executor._build_price_cache()
            self._isolated_env_cached = isolated_env
            self._isolated_executor_cached = isolated_executor
        else:
            isolated_env = self._isolated_env_cached
            isolated_executor = self._isolated_executor_cached
            self._reset_isolated_env_state(isolated_env, initial_capital)
        strategy_info = STRATEGY_ACTIONS[strategy_id]
        strategy_name = strategy_info.get('legacy_name', strategy_info['name'])
        canonical_strategy_name = strategy_info['name']
        adaptive_params = isolated_executor._get_adaptive_parameters(strategy_name)
        try:
            if strategy_name == 'MACross':
                result = isolated_executor._execute_ma_cross_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'Momentum':
                result = isolated_executor._execute_momentum_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'Turtle':
                result = isolated_executor._execute_turtle_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'Volume':
                result = isolated_executor._execute_volume_breakout_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'ATR':
                result = isolated_executor._execute_atr_breakout_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'Boll':
                result = isolated_executor._execute_boll_reversion_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'RSI':
                result = isolated_executor._execute_rsi_reversion_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'KDJ':
                result = isolated_executor._execute_kdj_reversion_adaptive(start_step, end_step, adaptive_params)
            elif strategy_name == 'LongOnly':
                result = isolated_executor._execute_long_only(start_step, end_step)
            elif strategy_name == 'ShortOnly':
                result = isolated_executor._execute_short_only(start_step, end_step)
            elif strategy_name == 'Cash':
                result = isolated_executor._execute_cash_only(start_step, end_step)
            else:
                raise ValueError(f'error: {strategy_name}')
            result['final_balance'] = isolated_env.balance
            result['final_shares'] = isolated_env.shares_held
            max_step_len = len(isolated_executor.df) if isolated_executor.df is not None else len(isolated_env.prices)
            if isolated_env.shares_held > 0 and end_step > start_step:
                try:
                    last_price = isolated_executor._price_at(min(end_step - 1, max_step_len - 1))
                    correct_portfolio_value = isolated_env.balance + isolated_env.shares_held * last_price
                    result['final_portfolio_value'] = correct_portfolio_value
                    result['mark_to_market_applied'] = True
                except Exception as e:
                    print(f'Warning: Failed to apply mark-to-market: {e}')
                    result['final_portfolio_value'] = isolated_env.portfolio_value
                    result['mark_to_market_applied'] = False
            elif hasattr(isolated_env, 'short_shares') and isolated_env.short_shares > 0 and (end_step > start_step):
                try:
                    last_price = isolated_executor._price_at(min(end_step - 1, max_step_len - 1))
                    short_pnl = isolated_env.short_shares * (isolated_env.short_entry_price - last_price)
                    correct_portfolio_value = isolated_env.balance + short_pnl
                    result['final_portfolio_value'] = correct_portfolio_value
                    result['mark_to_market_applied'] = True
                except Exception as e:
                    print(f'Warning: Failed to apply mark-to-market for short: {e}')
                    result['final_portfolio_value'] = isolated_env.portfolio_value
                    result['mark_to_market_applied'] = False
            else:
                result['final_portfolio_value'] = isolated_env.portfolio_value
                result['mark_to_market_applied'] = False
            result['adaptive_params'] = adaptive_params
            result['strategy_name'] = canonical_strategy_name
            result['debug_trades'] = isolated_env.trades
            result['debug_equity_curve'] = isolated_env.daily_equity_curve
            result['debug_start_price'] = isolated_executor._price_at(start_step) if start_step < max_step_len else 0
            result['debug_end_price'] = isolated_executor._price_at(min(end_step - 1, max_step_len - 1)) if end_step <= max_step_len else 0
            return result
        except Exception as e:
            print(f' {strategy_name} error: {e}')
            return {'total_return': 0, 'buy_hold_return': 0, 'excess_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'num_trades': 0, 'final_balance': initial_capital, 'final_shares': 0, 'final_portfolio_value': initial_capital, 'adaptive_params': adaptive_params, 'strategy_name': canonical_strategy_name, 'error': str(e)}

    def execute_strategy(self, strategy_id, start_step, end_step, initial_capital=None):
        if initial_capital is not None:
            return self.execute_strategy_isolated(strategy_id, start_step, end_step, initial_capital)
        else:
            return self._execute_strategy_original(strategy_id, start_step, end_step)

    def _execute_strategy_original(self, strategy_id, start_step, end_step):
        original_balance = self.env.balance
        original_shares = self.env.shares_held
        original_portfolio_value = self.env.portfolio_value
        original_initial_balance = self.env.initial_balance
        self.env.shares_held = 0
        self.env.short_shares = 0
        self.env.trades = []
        self.env.daily_equity_curve = []
        strategy_info = STRATEGY_ACTIONS[strategy_id]
        strategy_name = strategy_info.get('legacy_name', strategy_info['name'])
        canonical_strategy_name = strategy_info['name']
        adaptive_params = self._get_adaptive_parameters(strategy_name)
        if strategy_name == 'MACross':
            result = self._execute_ma_cross_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'Momentum':
            result = self._execute_momentum_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'Turtle':
            result = self._execute_turtle_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'Volume':
            result = self._execute_volume_breakout_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'ATR':
            result = self._execute_atr_breakout_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'Boll':
            result = self._execute_boll_reversion_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'RSI':
            result = self._execute_rsi_reversion_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'KDJ':
            result = self._execute_kdj_reversion_adaptive(start_step, end_step, adaptive_params)
        elif strategy_name == 'LongOnly':
            result = self._execute_long_only(start_step, end_step)
        elif strategy_name == 'ShortOnly':
            result = self._execute_short_only(start_step, end_step)
        elif strategy_name == 'Cash':
            result = self._execute_cash_only(start_step, end_step)
        else:
            raise ValueError(f'error: {strategy_name}')
        final_balance = self.env.balance
        final_shares = self.env.shares_held
        final_portfolio_value = self.env.portfolio_value
        self.env.balance = original_balance
        self.env.shares_held = original_shares
        self.env.portfolio_value = original_portfolio_value
        self.env.initial_balance = original_initial_balance
        result['final_balance'] = final_balance
        result['final_shares'] = final_shares
        result['final_portfolio_value'] = final_portfolio_value
        result['adaptive_params'] = adaptive_params
        result['strategy_name'] = canonical_strategy_name
        return result

    def _open_long_position(self, position_size=None):
        if self.env.current_price <= 0:
            return
        if position_size is None:
            position_size = self._calculate_dynamic_position_size('long')
        available_cash = self.env.balance * position_size
        max_shares_before_fee = available_cash / self.env.current_price
        max_shares = int(max_shares_before_fee / (1 + COMMISSION_RATE))
        if max_shares > 0:
            self.env.shares_held = max_shares
            cost = max_shares * self.env.current_price * (1 + COMMISSION_RATE)
            self.env.balance -= cost
            self.env.portfolio_value = self.env.balance + self.env.shares_held * self.env.current_price
            self.env.trades.append({'type': 'buy', 'step': self.env.current_step, 'price': self.env.current_price, 'shares': max_shares, 'cost': cost, 'portfolio_value': self.env.portfolio_value, 'position_size': position_size})

    def _open_short_position(self, position_size=None):
        if not getattr(self.env, 'allow_short', True):
            return
        if self.env.current_price <= 0:
            return
        if position_size is None:
            position_size = self._calculate_dynamic_position_size('short')
        current_equity = self.env.balance + self.env.shares_held * self.env.current_price
        if hasattr(self.env, 'short_shares') and self.env.short_shares > 0:
            current_equity = self.env.balance - self.env.short_shares * self.env.current_price
        target_short_notional = current_equity * position_size
        max_shares = int(target_short_notional / (self.env.current_price * (1 + COMMISSION_RATE)))
        if max_shares > 0:
            self.env.short_shares = max_shares
            self.env.short_entry_price = self.env.current_price
            proceeds = max_shares * self.env.current_price * (1 - COMMISSION_RATE)
            self.env.balance += proceeds
            self.env.portfolio_value = self.env.balance - self.env.short_shares * self.env.current_price
            self.env.trades.append({'type': 'short', 'step': self.env.current_step, 'price': self.env.current_price, 'shares': max_shares, 'proceeds': proceeds, 'portfolio_value': self.env.portfolio_value, 'position_size': position_size})

    def _add_to_position(self, position_size):
        if self.env.current_price <= 0:
            return
        available_cash = self.env.balance * position_size
        additional_shares_before_fee = available_cash / self.env.current_price
        additional_shares = int(additional_shares_before_fee / (1 + self.env.commission_rate))
        if additional_shares > 0:
            self.env.shares_held += additional_shares
            cost = additional_shares * self.env.current_price * (1 + self.env.commission_rate)
            self.env.balance -= cost
            self.env.portfolio_value = self.env.balance + self.env.shares_held * self.env.current_price
            self.env.trades.append({'type': 'add', 'step': self.env.current_step, 'price': self.env.current_price, 'shares': additional_shares, 'cost': cost, 'portfolio_value': self.env.portfolio_value, 'position_size': position_size})

    def _execute_long_only(self, start_step, end_step):
        trades = []
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        position_size = self._calculate_dynamic_position_size('long')
        position_size = max(0.85, position_size)
        self._open_long_position(position_size)
        trades.append({'step': start_step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'long_only_entry_{position_size:.1%}', 'shares': self.env.shares_held})
        self._update_daily_equity()
        last_adjustment_step = start_step
        max_step = len(self.df) if self.df is not None else len(self.env.prices)
        for step in range(start_step + 1, min(end_step, max_step)):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step < end_step - 1 and step - last_adjustment_step >= 20:
                new_position_size = self._calculate_dynamic_position_size('long')
                new_position_size = max(0.85, new_position_size)
                current_position_value = self.env.shares_held * self.env.current_price
                total_value = self.env.balance + current_position_value
                current_position_size = current_position_value / total_value if total_value > 0 else 0
                if abs(new_position_size - current_position_size) > 0.08:
                    self._close_long_position()
                    self._open_long_position(new_position_size)
                    trades.append({'step': step, 'action': 'rebalance', 'price': self.env.current_price, 'signal': f'position_adjust_{current_position_size:.1%}_to_{new_position_size:.1%}', 'shares': self.env.shares_held})
                    last_adjustment_step = step
            self._update_daily_equity()
        final_step = min(end_step - 1, max_step - 1)
        if self.env.current_step < final_step:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._update_daily_equity()
        self._close_long_position()
        trades.append({'step': self.env.current_step, 'action': 'sell', 'price': self.env.current_price, 'signal': 'long_only_exit', 'shares': 0})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_short_only(self, start_step, end_step):
        if not getattr(self.env, 'allow_short', True):
            return self._execute_cash_only(start_step, end_step)
        trades = []
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        position_size = self._calculate_dynamic_position_size('short')
        position_size = max(0.85, position_size)
        self._open_short_position(position_size)
        trades.append({'step': start_step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'short_only_entry_{position_size:.1%}', 'shares': self.env.short_shares})
        self._update_daily_equity()
        last_adjustment_step = start_step
        max_step = len(self.df) if self.df is not None else len(self.env.prices)
        for step in range(start_step + 1, min(end_step, max_step)):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step < end_step - 1 and step - last_adjustment_step >= 20:
                new_position_size = self._calculate_dynamic_position_size('short')
                new_position_size = max(0.85, new_position_size)
                if hasattr(self.env, 'short_shares') and self.env.short_shares > 0:
                    short_notional = self.env.short_shares * self.env.current_price
                    current_equity = self.env.balance - short_notional
                    current_position_size = short_notional / current_equity if current_equity > 0 else 0
                    if abs(new_position_size - current_position_size) > 0.08:
                        self._close_short_position()
                        self._open_short_position(new_position_size)
                        trades.append({'step': step, 'action': 'rebalance', 'price': self.env.current_price, 'signal': f'position_adjust_{current_position_size:.1%}_to_{new_position_size:.1%}', 'shares': self.env.short_shares})
                        last_adjustment_step = step
            self._update_daily_equity()
        final_step = min(end_step - 1, max_step - 1)
        if self.env.current_step < final_step:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._update_daily_equity()
        self._close_short_position()
        trades.append({'step': self.env.current_step, 'action': 'cover', 'price': self.env.current_price, 'signal': 'short_only_exit', 'shares': 0})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_cash_only(self, start_step, end_step):
        trades = []
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        max_step = len(self.df) if self.df is not None else len(self.env.prices)
        for step in range(start_step + 1, min(end_step, max_step)):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_ma_cross_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'MACross')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'MACross_{reason}')
        fast_period = self._get_adaptive_period(window_size, params.get('fast_period', 5), min_period=2, max_ratio=0.25)
        slow_period = self._get_adaptive_period(window_size, params.get('slow_period', 20), min_period=3, max_ratio=0.4)
        if fast_period >= slow_period:
            slow_period = min(fast_period + 2, int(window_size * 0.4))
            if slow_period <= fast_period:
                return self._execute_fallback_strategy(start_step, end_step, f'MACross_period_conflict_f{fast_period}_s{slow_period}')
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        trades = []
        position = 0
        entry_prices = []
        position_sizes = []
        max_position_layers = 4 if allow_pyramid else 1
        trades = []
        position = 0
        entry_prices = []
        position_sizes = []
        max_position_layers = 4 if allow_pyramid else 1
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        prices_all = precomputed_indicators['CLOSING_PRICES']
        ma_fast_values = precomputed_indicators['MA10_VALUES'] if fast_period == 10 else self._calculate_ma_series(prices_all, fast_period)
        ma_slow_values = precomputed_indicators['MA20_VALUES'] if slow_period == 20 else self._calculate_ma_series(prices_all, slow_period)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            current_indicators = precomputed_indicators
            if step - start_step < slow_period:
                if step > start_step:
                    self._update_daily_equity()
                continue
            if step >= 5:
                trend_strength = (self.env.current_price - self._price_at(step - 5)) / self._price_at(step - 5)
            else:
                trend_strength = 0
            if position != 0 and entry_prices:
                total_cost = sum((self._price_at(entry_prices[i]) * position_sizes[i] for i in range(len(entry_prices))))
                total_shares = sum(position_sizes)
                avg_entry_price = total_cost / total_shares if total_shares > 0 else 0
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                if position > 0:
                    base_atr_mult = params.get('atr_multiplier', 1.5)
                    if trend_strength > 0.05:
                        atr_mult = base_atr_mult * 1.5
                    elif trend_strength > 0.02:
                        atr_mult = base_atr_mult * 1.2
                    else:
                        atr_mult = base_atr_mult
                    stop_price = avg_entry_price - atr_mult * atr
                    if self.env.current_price > avg_entry_price * 1.03:
                        entry_step = min(entry_prices)
                        highest_since_entry = max((self._price_at(s) for s in range(entry_step, step + 1)))
                        profit_pct = (self.env.current_price - avg_entry_price) / avg_entry_price
                        if profit_pct > 0.15:
                            trailing_stop = highest_since_entry * 0.95
                        elif profit_pct > 0.08:
                            trailing_stop = highest_since_entry * 0.92
                        else:
                            trailing_stop = highest_since_entry - 1.2 * atr
                        stop_price = max(stop_price, trailing_stop)
                    if self.env.current_price < stop_price and trend_strength < -0.02:
                        self._close_long_position()
                        position = 0
                        position_sizes = []
                        entry_prices = []
                        trades.append({'step': step, 'action': 'stop_loss', 'price': self.env.current_price, 'signal': f'ma_stop_trend{trend_strength:.2%}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if len(ma_fast_values) > step and len(ma_slow_values) > step:
                current_ma_fast = ma_fast_values[step]
                current_ma_slow = ma_slow_values[step]
                prev_ma_fast = ma_fast_values[step - 1] if step > 0 else current_ma_fast
                prev_ma_slow = ma_slow_values[step - 1] if step > 0 else current_ma_slow
                ma_fast_slope = (current_ma_fast - prev_ma_fast) / prev_ma_fast if prev_ma_fast > 0 else 0
                ma_slow_slope = (current_ma_slow - prev_ma_slow) / prev_ma_slow if prev_ma_slow > 0 else 0
                if position <= 0:
                    if prev_ma_fast <= prev_ma_slow and current_ma_fast > current_ma_slow or (current_ma_fast > current_ma_slow and ma_fast_slope > 0.001 and (ma_slow_slope > 0)):
                        if position < 0:
                            self._close_short_position()
                            position = 0
                            position_sizes = []
                            entry_prices = []
                        signal_strength = abs(current_ma_fast - current_ma_slow) / current_ma_slow
                        if signal_strength > 0.02 or trend_strength > 0.03:
                            initial_position = 0.85
                        else:
                            initial_position = 0.7
                        self._open_long_position(initial_position)
                        position = 1
                        position_sizes.append(self.env.shares_held)
                        entry_prices.append(step)
                        trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'ma_cross_buy_{initial_position:.1%}', 'shares': self.env.shares_held})
                elif position > 0 and len(position_sizes) < max_position_layers and allow_pyramid:
                    should_add = False
                    add_reason = ''
                    if step >= 5:
                        recent_high = max((self._price_at(s) for s in range(step - 5, step)))
                        if self.env.current_price > recent_high * 1.005:
                            should_add = True
                            add_reason = 'breakout'
                    ma_spread = (current_ma_fast - current_ma_slow) / current_ma_slow
                    if ma_spread > 0.01 * len(position_sizes):
                        should_add = True
                        add_reason = 'ma_diverge'
                    if len(entry_prices) > 0 and step >= 5:
                        last_entry_price = self._price_at(entry_prices[-1])
                        recent_high = max((self._price_at(s) for s in range(max(start_step, step - 5), step)))
                        if self.env.current_price > last_entry_price * 1.01 and self.env.current_price < recent_high * 0.99:
                            should_add = True
                            add_reason = 'pullback'
                    if should_add:
                        add_size = 0.15 + 0.05 * len(position_sizes)
                        prev_shares = self.env.shares_held
                        self._add_to_position(add_size)
                        added_shares = self.env.shares_held - prev_shares
                        position_sizes.append(added_shares)
                        entry_prices.append(step)
                        trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'ma_add_{add_reason}_layer{len(position_sizes)}', 'shares': added_shares})
                elif position >= 0 and prev_ma_fast >= prev_ma_slow and (current_ma_fast < current_ma_slow):
                    if trend_strength < -0.02 and ma_fast_slope < -0.001:
                        if position > 0:
                            self._close_long_position()
                            trades.append({'step': step, 'action': 'close_long', 'price': self.env.current_price, 'signal': f'ma_death_cross'})
                        if not is_short_window:
                            self._open_short_position(0.7)
                            position = -1
                            position_sizes = [0.7]
                            entry_prices = [step]
                            trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'ma_cross_sell', 'shares': self.env.shares_held})
                        else:
                            position = 0
                            position_sizes = []
                            entry_prices = []
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_momentum_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'Momentum')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'Momentum_{reason}')
        lookback = self._get_adaptive_period(window_size, params.get('lookback', 10), min_period=3, max_ratio=0.3)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        momentum_history = []
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < max(lookback, 14):
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            momentum_short = momentum_mid = momentum_long = 0
            if step >= 5:
                momentum_short = (self.env.current_price - self._price_at(step - 5)) / self._price_at(step - 5)
            if step >= lookback:
                current_price = current_indicators['CLOSING_PRICES'][step]
                lookback_price = current_indicators['CLOSING_PRICES'][step - lookback]
                if lookback_price > 0:
                    momentum_mid = (current_price - lookback_price) / lookback_price
            if step >= 20:
                momentum_long = (self.env.current_price - self._price_at(step - 20)) / self._price_at(step - 20)
            momentum = 0.5 * momentum_mid + 0.3 * momentum_short + 0.2 * momentum_long
            momentum_history.append(momentum)
            momentum_acceleration = 0
            if len(momentum_history) >= 5:
                recent_momentum = momentum_history[-5:]
                momentum_acceleration = recent_momentum[-1] - recent_momentum[0]
            base_entry = params.get('entry_threshold', 0.02)
            base_exit = params.get('exit_threshold', 0.005)
            if self.volatility < 0.15:
                entry_threshold = base_entry * 0.7
                exit_threshold = base_exit * 0.5
            elif self.volatility > 0.3:
                entry_threshold = base_entry * 1.2
                exit_threshold = base_exit * 1.5
            else:
                entry_threshold = base_entry
                exit_threshold = base_exit
            rsi_values = current_indicators['RSI_VALUES']
            current_rsi = rsi_values[step] if step < len(rsi_values) else 50
            volumes = current_indicators['VOLUMES']
            if step >= 5:
                avg_volume = np.mean(volumes[step - 5:step])
                current_volume = volumes[step] if step < len(volumes) else avg_volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_ratio = 1
            if position != 0 and entry_price > 0:
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                if position > 0:
                    base_atr_mult = params.get('atr_multiplier', 1.5)
                    holding_days = step - entry_prices[0] if entry_prices else 0
                    profit_pct = (self.env.current_price - entry_price) / entry_price
                    if profit_pct > 0.1:
                        atr_mult = base_atr_mult * 0.8
                    elif holding_days > 10 and profit_pct > 0:
                        atr_mult = base_atr_mult * 1.2
                    else:
                        atr_mult = base_atr_mult
                    stop_price = entry_price - atr_mult * atr
                    momentum_stop = momentum < -exit_threshold and momentum_acceleration < 0
                    if self.env.current_price > entry_price * 1.05:
                        profit_levels = [0.05, 0.08, 0.12, 0.2]
                        stop_levels = [0.97, 0.96, 0.94, 0.9]
                        for i, level in enumerate(profit_levels):
                            if profit_pct > level:
                                trailing_stop = entry_price * (1 + level) * stop_levels[i]
                                stop_price = max(stop_price, trailing_stop)
                    if (self.env.current_price < stop_price or momentum_stop) and position_layers > 0:
                        if position_layers > 1 and momentum > -exit_threshold * 2:
                            shares_to_sell = self.env.shares_held // 2
                            self.env.shares_held -= shares_to_sell
                            proceeds = shares_to_sell * self.env.current_price * (1 - COMMISSION_RATE)
                            self.env.balance += proceeds
                            position_layers -= 1
                            trades.append({'step': step, 'action': 'partial_close', 'price': self.env.current_price, 'signal': f'momentum_partial_{momentum:.2%}'})
                        else:
                            self._close_long_position()
                            position = 0
                            position_layers = 0
                            trades.append({'step': step, 'action': 'stop_loss', 'price': self.env.current_price, 'signal': f'momentum_stop_{momentum:.2%}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                long_signal = False
                signal_strength = 0
                if momentum > entry_threshold:
                    signal_strength += 1
                    if momentum_acceleration > 0:
                        signal_strength += 1
                    if volume_ratio > 1.2:
                        signal_strength += 1
                    if current_rsi > 50 and current_rsi < 80:
                        signal_strength += 1
                    long_signal = True
                elif len(momentum_history) >= 3:
                    if momentum_history[-3] < -entry_threshold and momentum > 0 and (momentum_acceleration > entry_threshold):
                        long_signal = True
                        signal_strength = 2
                if long_signal:
                    if signal_strength >= 3:
                        position_size = 0.9
                    elif signal_strength >= 2:
                        position_size = 0.8
                    else:
                        position_size = 0.7
                    self._open_long_position(position_size)
                    position = 1
                    position_layers = 1
                    entry_price = self.env.current_price
                    entry_prices = [step]
                    trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'momentum_{momentum:.2%}_strength{signal_strength}', 'shares': self.env.shares_held})
                elif momentum < -entry_threshold and momentum_acceleration < 0 and (current_rsi < 40) and (not is_short_window):
                    self._open_short_position(0.7)
                    position = -1
                    position_layers = 1
                    entry_price = self.env.current_price
                    entry_prices = [step]
                    trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'momentum_{momentum:.2%}', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                add_conditions = 0
                if momentum > entry_threshold * (1 + 0.5 * position_layers):
                    add_conditions += 1
                if self.env.current_price > entry_price * (1 + 0.01 * position_layers):
                    add_conditions += 1
                if momentum_acceleration > entry_threshold * 0.5:
                    add_conditions += 1
                if volume_ratio > 1.3:
                    add_conditions += 1
                if add_conditions >= 2:
                    add_size = 0.3 - 0.05 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'momentum_pyramid_{momentum:.2%}_layer{position_layers}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_turtle_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'Turtle')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'Turtle_{reason}')
        entry_window = self._get_adaptive_period(window_size, params.get('entry_window', 20), min_period=5, max_ratio=0.4)
        exit_window = self._get_adaptive_period(window_size, params.get('exit_window', 10), min_period=3, max_ratio=0.3)
        if exit_window >= entry_window:
            exit_window = max(3, entry_window // 2)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_units = 5 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_units = 0
        last_breakout_high = 0
        consecutive_wins = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_20 = self._calculate_atr(20)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < entry_window:
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            if self.volatility > 0.25:
                actual_entry_window = max(10, entry_window - 5)
                actual_exit_window = max(5, exit_window - 2)
            else:
                actual_entry_window = entry_window
                actual_exit_window = exit_window
            entry_high = max(prices[step - actual_entry_window:step])
            entry_low = min(prices[step - actual_entry_window:step])
            exit_high = max(prices[step - actual_exit_window:step]) if step >= actual_exit_window else entry_high
            exit_low = min(prices[step - actual_exit_window:step]) if step >= actual_exit_window else entry_low
            price_range = entry_high - entry_low
            breakout_strength = abs(self.env.current_price - entry_high) / price_range if price_range > 0 else 0
            atr = atr_values_20[step] if step < len(atr_values_20) else self.env.current_price * 0.02
            volumes = current_indicators['VOLUMES']
            if step >= 5:
                avg_volume = np.mean(volumes[step - 5:step])
                current_volume = volumes[step] if step < len(volumes) else avg_volume
                volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_surge = 1
            if position != 0 and entry_price > 0:
                base_atr_mult = params.get('atr_multiplier', 2.0)
                profit_pct = (self.env.current_price - entry_price) / entry_price if position > 0 else (entry_price - self.env.current_price) / entry_price
                if profit_pct > 0.1:
                    atr_mult = base_atr_mult * 0.7
                elif position_units >= 3:
                    atr_mult = base_atr_mult * 1.3
                else:
                    atr_mult = base_atr_mult
                if position > 0:
                    stop_price = entry_price - atr_mult * atr
                    if profit_pct > 0.15 and position_units > 1:
                        shares_to_sell = self.env.shares_held // 2
                        self.env.shares_held -= shares_to_sell
                        proceeds = shares_to_sell * self.env.current_price * (1 - COMMISSION_RATE)
                        self.env.balance += proceeds
                        position_units = max(1, position_units - 1)
                        trades.append({'step': step, 'action': 'partial_profit', 'price': self.env.current_price, 'signal': f'turtle_partial_profit_{profit_pct:.1%}'})
                    if self.env.current_price < stop_price or (self.env.current_price < exit_low and profit_pct < 0):
                        self._close_long_position()
                        position = 0
                        position_units = 0
                        if profit_pct > 0:
                            consecutive_wins += 1
                        else:
                            consecutive_wins = 0
                        trades.append({'step': step, 'action': 'stop_loss', 'price': self.env.current_price, 'signal': f'turtle_stop_{profit_pct:.1%}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                if self.env.current_price > entry_high:
                    quality_score = 0
                    if self.env.current_price > entry_high * 1.005:
                        quality_score += 1
                    if volume_surge > 1.3:
                        quality_score += 1
                    if self.env.current_price > entry_high * 1.002:
                        quality_score += 1
                    ma20 = np.mean(prices[step - 20:step]) if step >= 20 else self.env.current_price
                    if self.env.current_price > ma20:
                        quality_score += 1
                    if quality_score >= 3:
                        unit_size = min(0.5, self.env.balance * 0.02 / atr)
                    elif quality_score >= 2:
                        unit_size = min(0.4, self.env.balance * 0.015 / atr)
                    else:
                        unit_size = min(0.3, self.env.balance * 0.01 / atr)
                    if consecutive_wins >= 2:
                        unit_size *= 1.2
                    self._open_long_position(unit_size)
                    position = 1
                    position_units = 1
                    entry_price = self.env.current_price
                    entry_prices = [step]
                    last_breakout_high = entry_high
                    trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'turtle_breakout_quality{quality_score}', 'shares': self.env.shares_held})
                elif self.env.current_price < entry_low and volume_surge > 1.5 and (not is_short_window):
                    ma20 = np.mean(prices[step - 20:step]) if step >= 20 else self.env.current_price
                    if self.env.current_price < ma20 * 0.98:
                        unit_size = min(0.3, self.env.balance * 0.01 / atr)
                        self._open_short_position(unit_size)
                        position = -1
                        position_units = 1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'turtle_breakdown', 'shares': self.env.shares_held})
            elif position > 0 and position_units < max_units and allow_pyramid:
                pyramid_atr = params.get('pyramid_atr', 0.5)
                should_add = False
                if self.env.current_price > entry_price + position_units * pyramid_atr * atr:
                    should_add = True
                if step >= entry_window:
                    new_high = max(prices[step - entry_window:step])
                    if self.env.current_price > new_high and new_high > last_breakout_high:
                        should_add = True
                        last_breakout_high = new_high
                if position_units < 3:
                    recent_low = min(prices[step - 5:step]) if step >= 5 else self.env.current_price
                    if recent_low > entry_price * (1 + 0.005 * position_units):
                        should_add = True
                if should_add:
                    unit_multiplier = 1 + 0.1 * position_units
                    unit_size = min(0.25 * unit_multiplier, self.env.balance * 0.015 / atr)
                    self._add_to_position(unit_size)
                    position_units += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'turtle_pyramid_unit{position_units}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_volume_breakout_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'Volume')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'Volume_{reason}')
        price_window = self._get_adaptive_period(window_size, params.get('price_window', 20), min_period=5, max_ratio=0.4)
        volume_window = self._get_adaptive_period(window_size, params.get('volume_window', 5), min_period=2, max_ratio=0.25)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        breakout_count = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < price_window:
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            volumes = current_indicators['VOLUMES']
            actual_index = len(prices) - 1
            if self.volatility > 0.25:
                actual_window = max(10, price_window - 5)
            elif self.volatility < 0.15:
                actual_window = min(30, price_window + 5)
            else:
                actual_window = price_window
            window_start = max(start_step, step - actual_window)
            window_prices = [self._price_at(s) for s in range(window_start, step)]
            if len(window_prices) > 0:
                recent_high = max(window_prices)
                recent_low = min(window_prices)
            else:
                recent_high = recent_low = self.env.current_price
            price_range = recent_high - recent_low
            if step >= volume_window:
                avg_volume_short = np.mean(volumes[actual_index - volume_window:actual_index])
                current_volume = volumes[actual_index] if actual_index < len(volumes) else avg_volume_short
                volume_ratio_short = current_volume / avg_volume_short if avg_volume_short > 0 else 1
                if step >= 20:
                    avg_volume_long = np.mean(volumes[actual_index - 20:actual_index])
                    volume_ratio_long = current_volume / avg_volume_long if avg_volume_long > 0 else 1
                else:
                    volume_ratio_long = volume_ratio_short
                volume_trend = 0
                if step >= volume_window + 5:
                    recent_avg = np.mean(volumes[actual_index - volume_window:actual_index])
                    past_avg = np.mean(volumes[actual_index - volume_window - 5:actual_index - 5])
                    if past_avg > 0:
                        volume_trend = (recent_avg - past_avg) / past_avg
            else:
                volume_ratio_short = volume_ratio_long = 1
                volume_trend = 0
            ma_period = params.get('ma_period', 10)
            if step >= ma_period:
                ma_value = np.mean(prices[actual_index - ma_period:actual_index])
                ma_slope = (ma_value - np.mean(prices[actual_index - ma_period - 1:actual_index - 1])) / ma_value if step > ma_period else 0
            else:
                ma_value = self.env.current_price
                ma_slope = 0
            if position != 0 and entry_price > 0:
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                base_atr_mult = params.get('atr_multiplier', 1.8)
                if position > 0:
                    profit_pct = (self.env.current_price - entry_price) / entry_price
                    if profit_pct > 0.08:
                        atr_mult = base_atr_mult * 0.8
                    elif breakout_count >= 2:
                        atr_mult = base_atr_mult * 1.2
                    else:
                        atr_mult = base_atr_mult
                    stop_price = entry_price - atr_mult * atr
                    if profit_pct > 0.05:
                        trailing_stop = self.env.current_price * (1 - min(0.04, profit_pct * 0.3))
                        stop_price = max(stop_price, trailing_stop)
                    volume_exhaustion = volume_ratio_short < 0.5 and self.env.current_price < ma_value
                    if self.env.current_price < stop_price or volume_exhaustion:
                        self._close_long_position()
                        position = 0
                        position_layers = 0
                        breakout_count = 0
                        trades.append({'step': step, 'action': 'stop_loss', 'price': self.env.current_price, 'signal': f"volume_stop_{('exhaustion' if volume_exhaustion else 'price')}"})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                breakout_score = 0
                signal_details = []
                volume_multiplier = params.get('volume_multiplier', 1.5)
                if self.env.current_price > recent_high:
                    price_breakout_pct = (self.env.current_price - recent_high) / recent_high
                    if price_breakout_pct > 0.002:
                        breakout_score += 2
                        signal_details.append('price')
                    if volume_ratio_short > volume_multiplier * 0.8:
                        breakout_score += 2
                        signal_details.append('vol_short')
                    elif volume_ratio_long > volume_multiplier * 0.9:
                        breakout_score += 1
                        signal_details.append('vol_long')
                    if volume_trend > 0.1:
                        breakout_score += 1
                        signal_details.append('vol_trend')
                    if self.env.current_price > ma_value:
                        breakout_score += 1
                        signal_details.append('above_ma')
                    if ma_slope > 0.001:
                        breakout_score += 1
                        signal_details.append('ma_up')
                    if breakout_score >= 3:
                        if breakout_score >= 6:
                            position_size = 0.9
                        elif breakout_score >= 4:
                            position_size = 0.8
                        else:
                            position_size = 0.7
                        self._open_long_position(position_size)
                        position = 1
                        position_layers = 1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        breakout_count = 1
                        trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f"vol_break_{'_'.join(signal_details[:3])}_score{breakout_score}", 'shares': self.env.shares_held})
                elif self.env.current_price < recent_low and volume_ratio_short > volume_multiplier and (not is_short_window):
                    if self.env.current_price < ma_value * 0.98 and ma_slope < -0.001:
                        self._open_short_position(0.7)
                        position = -1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'vol_breakdown', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                new_breakout = False
                if self.env.current_price > recent_high * 1.01 and volume_ratio_short > 1.2:
                    new_breakout = True
                    breakout_count += 1
                strong_trend = self.env.current_price > entry_price * (1 + 0.02 * position_layers) and ma_slope > 0.002 and (volume_trend > 0)
                if new_breakout or (strong_trend and self.env.current_price > ma_value * 1.01):
                    add_size = 0.25 + 0.05 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f"vol_add_{('new_break' if new_breakout else 'trend')}_layer{position_layers}", 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_atr_breakout_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'ATR')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'ATR_{reason}')
        atr_period = self._get_adaptive_period(window_size, params.get('atr_period', 20), min_period=5, max_ratio=0.4)
        ma_period = self._get_adaptive_period(window_size, params.get('ma_period', 20), min_period=5, max_ratio=0.4)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        channel_breaks = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_period = self._calculate_atr(atr_period)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < max(atr_period, ma_period):
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            ma_value = np.mean(prices[step - ma_period:step])
            atr = atr_values_period[step] if step < len(atr_values_period) else self.env.current_price * 0.02
            momentum_5 = (self.env.current_price - prices[step - 5]) / prices[step - 5] if step >= 5 else 0
            momentum_10 = (self.env.current_price - prices[step - 10]) / prices[step - 10] if step >= 10 else 0
            base_entry_mult = params.get('atr_mult_entry', 1.5)
            if self.volatility < 0.15:
                atr_mult_entry = base_entry_mult * 0.8
            elif self.volatility > 0.3:
                atr_mult_entry = base_entry_mult * 1.2
            else:
                atr_mult_entry = base_entry_mult
            if abs(momentum_10) > 0.05:
                atr_mult_entry *= 0.9
            atr_mult_exit = params.get('atr_mult_exit', 0.75)
            atr_mult_stop = params.get('atr_mult_stop', 1.2)
            upper_band = ma_value + atr_mult_entry * atr
            lower_band = ma_value - atr_mult_entry * atr
            exit_upper = ma_value + atr_mult_exit * atr
            exit_lower = ma_value - atr_mult_exit * atr
            if step >= atr_period + 5:
                prev_atr = atr_values_period[step - 5] if step - 5 < len(atr_values_period) else atr
                channel_expanding = atr > prev_atr * 1.1
                channel_contracting = atr < prev_atr * 0.9
            else:
                channel_expanding = channel_contracting = False
            if position != 0 and entry_price > 0:
                profit_pct = (self.env.current_price - entry_price) / entry_price if position > 0 else (entry_price - self.env.current_price) / entry_price
                if position > 0:
                    if profit_pct > 0.1:
                        stop_mult = atr_mult_stop * 0.8
                    elif channel_expanding:
                        stop_mult = atr_mult_stop * 1.3
                    else:
                        stop_mult = atr_mult_stop
                    stop_price = entry_price - stop_mult * atr
                    if channel_contracting and profit_pct > 0.05:
                        exit_price = ma_value + atr_mult_exit * 0.5 * atr
                    else:
                        exit_price = exit_lower
                    if profit_pct > 0.06:
                        trailing_stop = self.env.current_price * (1 - min(0.04, profit_pct * 0.25))
                        stop_price = max(stop_price, trailing_stop)
                    if self.env.current_price < stop_price or (self.env.current_price < exit_price and momentum_5 < -0.01):
                        self._close_long_position()
                        position = 0
                        position_layers = 0
                        channel_breaks = 0
                        trades.append({'step': step, 'action': 'stop_loss', 'price': self.env.current_price, 'signal': f'atr_stop_{profit_pct:.1%}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                if self.env.current_price > upper_band:
                    quality_score = 0
                    breakout_pct = (self.env.current_price - upper_band) / upper_band
                    if breakout_pct > 0.003:
                        quality_score += 2
                    elif breakout_pct > 0.001:
                        quality_score += 1
                    if momentum_5 > 0.02:
                        quality_score += 2
                    elif momentum_5 > 0.01:
                        quality_score += 1
                    if channel_expanding:
                        quality_score += 1
                    if ma_value > prices[step - 20] if step >= 20 else True:
                        quality_score += 1
                    if step >= 3:
                        recent_closes = prices[step - 3:step]
                        if all((close > ma_value for close in recent_closes)):
                            quality_score += 1
                    if quality_score >= 3:
                        if quality_score >= 6:
                            position_size = 0.9
                        elif quality_score >= 4:
                            position_size = 0.85
                        else:
                            position_size = 0.75
                        self._open_long_position(position_size)
                        position = 1
                        position_layers = 1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        channel_breaks = 1
                        trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'atr_breakout_up_q{quality_score}', 'shares': self.env.shares_held})
                elif self.env.current_price < lower_band and momentum_10 < -0.03 and (not is_short_window):
                    if channel_expanding or self.env.current_price < ma_value * 0.97:
                        self._open_short_position(0.7)
                        position = -1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'atr_breakout_down', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                add_signal = False
                add_reason = ''
                if self.env.current_price > upper_band * (1 + 0.005 * channel_breaks):
                    add_signal = True
                    add_reason = 'new_break'
                    channel_breaks += 1
                elif position_layers == 1:
                    recent_low = min(prices[step - 5:step]) if step >= 5 else self.env.current_price
                    if recent_low > entry_price and self.env.current_price > recent_low * 1.01:
                        add_signal = True
                        add_reason = 'pullback_rally'
                elif self.env.current_price > entry_price * (1 + 0.03 * position_layers) and momentum_5 > 0.01:
                    add_signal = True
                    add_reason = 'strong_trend'
                if add_signal:
                    add_size = 0.2 + 0.1 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'atr_add_{add_reason}_layer{position_layers}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_boll_reversion_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'Boll')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'Boll_{reason}')
        window = self._get_adaptive_period(window_size, params.get('window', 20), min_period=5, max_ratio=0.4)
        base_num_std = params.get('num_std', 1.8)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        extreme_count = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        prices_all = precomputed_indicators['CLOSING_PRICES']
        if self.volatility < 0.15:
            num_std = base_num_std * 0.9
        elif self.volatility > 0.3:
            num_std = base_num_std * 1.1
        else:
            num_std = base_num_std
        boll_upper_all, boll_middle_all, boll_lower_all = self._calculate_bollinger_bands_adaptive(prices_all, window, num_std)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < window:
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            if step < len(boll_upper_all):
                upper_band = boll_upper_all[step]
                middle_band = boll_middle_all[step]
                lower_band = boll_lower_all[step]
            else:
                continue
            band_width = upper_band - lower_band
            if band_width > 0:
                price_position = (self.env.current_price - lower_band) / band_width
            else:
                price_position = 0.5
            if step >= window + 5:
                prev_width = boll_upper_all[step - 5] - boll_lower_all[step - 5]
                band_expanding = band_width > prev_width * 1.1
                band_contracting = band_width < prev_width * 0.9
            else:
                band_expanding = band_contracting = False
            rsi_values = current_indicators['RSI_VALUES']
            current_rsi = rsi_values[step] if step < len(rsi_values) else 50
            momentum_3 = (self.env.current_price - prices[step - 3]) / prices[step - 3] if step >= 3 else 0
            momentum_5 = (self.env.current_price - prices[step - 5]) / prices[step - 5] if step >= 5 else 0
            volumes = current_indicators['VOLUMES']
            if step >= 5:
                avg_volume = np.mean(volumes[step - 5:step])
                current_volume = volumes[step] if step < len(volumes) else avg_volume
                volume_spike = current_volume > avg_volume * 1.3
            else:
                volume_spike = False
            if position != 0 and entry_price > 0:
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                base_atr_mult = params.get('atr_multiplier', 1.8)
                if position > 0:
                    profit_pct = (self.env.current_price - entry_price) / entry_price
                    if band_contracting and profit_pct > 0.02:
                        atr_mult = base_atr_mult * 0.7
                    elif extreme_count >= 2:
                        atr_mult = base_atr_mult * 1.2
                    else:
                        atr_mult = base_atr_mult
                    stop_price = entry_price - atr_mult * atr
                    target_price = middle_band
                    if profit_pct > 0.03:
                        target_price = middle_band + (upper_band - middle_band) * 0.2
                    if profit_pct > 0.04:
                        trailing_stop = self.env.current_price * 0.97
                        stop_price = max(stop_price, trailing_stop)
                    should_close = False
                    close_reason = ''
                    if self.env.current_price >= target_price and momentum_3 < 0:
                        should_close = True
                        close_reason = 'target_reached'
                    elif self.env.current_price < stop_price:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif price_position > 0.8 and current_rsi > 65:
                        should_close = True
                        close_reason = 'overbought'
                    if should_close:
                        self._close_long_position()
                        position = 0
                        position_layers = 0
                        extreme_count = 0
                        trades.append({'step': step, 'action': 'close', 'price': self.env.current_price, 'signal': f'boll_{close_reason}_{profit_pct:.1%}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                if price_position < 0.15:
                    oversold_score = 0
                    if self.env.current_price < lower_band:
                        oversold_score += 2
                        extreme_count = 1
                    if current_rsi < 40:
                        oversold_score += 1
                    if current_rsi < 30:
                        oversold_score += 1
                    if momentum_3 < -0.02:
                        oversold_score += 1
                    if volume_spike and momentum_3 < 0:
                        oversold_score += 1
                    if band_expanding:
                        oversold_score += 1
                    if oversold_score >= 3:
                        if oversold_score >= 5:
                            position_size = 0.9
                        elif oversold_score >= 4:
                            position_size = 0.85
                        else:
                            position_size = 0.75
                        self._open_long_position(position_size)
                        position = 1
                        position_layers = 1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f'boll_oversold_s{oversold_score}_rsi{current_rsi:.0f}', 'shares': self.env.shares_held})
                elif price_position > 0.9 and current_rsi > 65 and (momentum_5 < 0.01) and (not is_short_window):
                    if band_contracting or (volume_spike and momentum_3 < 0):
                        self._open_short_position(0.7)
                        position = -1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'boll_overbought_rsi{current_rsi:.0f}', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                add_signal = False
                add_reason = ''
                if self.env.current_price < lower_band * (1 - 0.01 * extreme_count):
                    add_signal = True
                    add_reason = 'more_extreme'
                    extreme_count += 1
                elif position_layers == 1 and self.env.current_price > entry_price * 0.99:
                    recent_low = min(prices[step - 3:step]) if step >= 3 else self.env.current_price
                    if recent_low < lower_band and self.env.current_price > recent_low * 1.005:
                        add_signal = True
                        add_reason = 'false_break'
                elif step - trades[-1]['step'] > 5:
                    if price_position < 0.3 and current_rsi < 45:
                        add_signal = True
                        add_reason = 'time_add'
                if add_signal:
                    add_size = 0.2 + 0.05 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'boll_add_{add_reason}_layer{position_layers}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_rsi_reversion_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'RSI')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'RSI_{reason}')
        oversold = params.get('oversold', 35)
        overbought = params.get('overbought', 65)
        neutral = params.get('neutral', 50)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        divergence_count = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        rsi_history = []
        price_history = []
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < 14:
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            rsi_values = current_indicators['RSI_VALUES']
            current_rsi = rsi_values[step] if step < len(rsi_values) else 50
            rsi_history.append(current_rsi)
            price_history.append(self.env.current_price)
            if self.volatility > 0.25:
                actual_oversold = oversold - 5
                actual_overbought = overbought + 5
            elif self.volatility < 0.15:
                actual_oversold = oversold + 5
                actual_overbought = overbought - 5
            else:
                actual_oversold = oversold
                actual_overbought = overbought
            rsi_momentum = 0
            if len(rsi_history) >= 5:
                rsi_momentum = rsi_history[-1] - rsi_history[-5]
            bullish_divergence = False
            bearish_divergence = False
            if len(price_history) >= 10:
                recent_prices = price_history[-10:]
                recent_rsi = rsi_history[-10:]
                price_low_idx = recent_prices.index(min(recent_prices))
                if price_low_idx >= 3:
                    prev_low_idx = recent_prices[:price_low_idx].index(min(recent_prices[:price_low_idx]))
                    if recent_prices[price_low_idx] < recent_prices[prev_low_idx] * 0.995:
                        if recent_rsi[price_low_idx] > recent_rsi[prev_low_idx]:
                            bullish_divergence = True
                            divergence_count += 1
            short_trend = (prices[step] - prices[step - 5]) / prices[step - 5] if step >= 5 else 0
            medium_trend = (prices[step] - prices[step - 10]) / prices[step - 10] if step >= 10 else 0
            volumes = current_indicators['VOLUMES']
            if step >= 5:
                avg_volume = np.mean(volumes[step - 5:step])
                current_volume = volumes[step] if step < len(volumes) else avg_volume
                volume_spike = current_volume > avg_volume * 1.3
            else:
                volume_spike = False
            if position != 0 and entry_price > 0:
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                base_atr_mult = params.get('atr_multiplier', 1.8)
                if position > 0:
                    profit_pct = (self.env.current_price - entry_price) / entry_price
                    if divergence_count >= 2:
                        atr_mult = base_atr_mult * 1.3
                    elif profit_pct > 0.05:
                        atr_mult = base_atr_mult * 0.8
                    else:
                        atr_mult = base_atr_mult
                    stop_price = entry_price - atr_mult * atr
                    if current_rsi > neutral:
                        rsi_exit_level = neutral + 5
                    else:
                        rsi_exit_level = neutral
                    if profit_pct > 0.04:
                        trailing_stop = self.env.current_price * 0.97
                        stop_price = max(stop_price, trailing_stop)
                    should_close = False
                    close_reason = ''
                    if current_rsi > rsi_exit_level and rsi_momentum < 0:
                        should_close = True
                        close_reason = 'rsi_reversion'
                    elif self.env.current_price < stop_price:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif profit_pct > 0.08 and current_rsi > 60:
                        should_close = True
                        close_reason = 'profit_target'
                    if should_close:
                        self._close_long_position()
                        position = 0
                        position_layers = 0
                        divergence_count = 0
                        trades.append({'step': step, 'action': 'close', 'price': self.env.current_price, 'signal': f'{close_reason}_rsi{current_rsi:.0f}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                oversold_score = 0
                if current_rsi < actual_oversold:
                    oversold_score += 2
                elif current_rsi < actual_oversold + 5:
                    oversold_score += 1
                if rsi_momentum < -10:
                    oversold_score += 1
                if short_trend < -0.02:
                    oversold_score += 1
                elif short_trend > 0.02:
                    oversold_score -= 1
                if bullish_divergence:
                    oversold_score += 2
                if volume_spike and short_trend < 0:
                    oversold_score += 1
                if current_rsi < 25:
                    oversold_score += 1
                if oversold_score >= 3:
                    if oversold_score >= 5:
                        position_size = 0.9
                    elif oversold_score >= 4:
                        position_size = 0.85
                    else:
                        position_size = 0.75
                    self._open_long_position(position_size)
                    position = 1
                    position_layers = 1
                    entry_price = self.env.current_price
                    entry_prices = [step]
                    trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f"rsi_oversold_{current_rsi:.0f}_s{oversold_score}{('_div' if bullish_divergence else '')}", 'shares': self.env.shares_held})
                elif current_rsi > actual_overbought and short_trend < 0.01 and (not is_short_window):
                    if rsi_momentum < -5 or (volume_spike and short_trend < 0):
                        self._open_short_position(0.7)
                        position = -1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'rsi_overbought_{current_rsi:.0f}', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                add_signal = False
                add_reason = ''
                if current_rsi < actual_oversold - 5 * position_layers:
                    add_signal = True
                    add_reason = 'rsi_extreme'
                elif self.env.current_price < entry_price * (1 - 0.02 * position_layers) and current_rsi < 45:
                    add_signal = True
                    add_reason = 'price_drop'
                elif bullish_divergence and divergence_count > 1:
                    add_signal = True
                    add_reason = 'new_divergence'
                elif position_layers == 1 and step - trades[-1]['step'] > 3:
                    if current_rsi < actual_oversold + 5:
                        add_signal = True
                        add_reason = 'time_add'
                if add_signal:
                    add_size = 0.2 + 0.05 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'rsi_add_{add_reason}_layer{position_layers}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _execute_kdj_reversion_adaptive(self, start_step, end_step, params):
        window_size = end_step - start_step
        should_fallback, reason = self._should_use_fallback_strategy(start_step, end_step, 'KDJ')
        if should_fallback:
            return self._execute_fallback_strategy(start_step, end_step, f'KDJ_{reason}')
        period = self._get_adaptive_period(window_size, params.get('period', 9), min_period=3, max_ratio=0.3)
        j_buy_threshold = params.get('j_buy_threshold', 10)
        k_buy_threshold = params.get('k_buy_threshold', 20)
        j_sell_threshold = params.get('j_sell_threshold', 90)
        k_sell_threshold = params.get('k_sell_threshold', 80)
        is_short_window = window_size < 20
        allow_pyramid = not is_short_window
        max_layers = 3 if allow_pyramid else 1
        trades = []
        position = 0
        entry_price = 0
        entry_prices = []
        position_layers = 0
        golden_cross_count = 0
        self.env.current_step = start_step
        self.env.current_price = self._price_at(start_step)
        self._update_daily_equity()
        kdj_history = {'k': [], 'd': [], 'j': []}
        loop_end = min(end_step, len(self.env.prices))
        precomputed_indicators = self._get_current_indicators(start_step, max(start_step, loop_end - 1))
        atr_values_14 = self._calculate_atr(14)
        for step in range(start_step, loop_end):
            self.env.current_step = step
            self.env.current_price = self._price_at(step)
            if step - start_step < period:
                if step > start_step:
                    self._update_daily_equity()
                continue
            current_indicators = precomputed_indicators
            prices = current_indicators['CLOSING_PRICES']
            k_values = current_indicators['KDJ_K_VALUES']
            d_values = current_indicators['KDJ_D_VALUES']
            j_values = current_indicators['KDJ_J_VALUES']
            if step < len(k_values):
                k_value = k_values[step]
                d_value = d_values[step]
                j_value = j_values[step]
            else:
                k_value = d_value = j_value = 50
            kdj_history['k'].append(k_value)
            kdj_history['d'].append(d_value)
            kdj_history['j'].append(j_value)
            if len(kdj_history['j']) >= 5:
                j_momentum = kdj_history['j'][-1] - kdj_history['j'][-5]
                k_momentum = kdj_history['k'][-1] - kdj_history['k'][-5]
            else:
                j_momentum = k_momentum = 0
            golden_cross = False
            death_cross = False
            if len(kdj_history['k']) >= 2:
                prev_k = kdj_history['k'][-2]
                prev_d = kdj_history['d'][-2]
                if prev_k <= prev_d and k_value > d_value:
                    golden_cross = True
                    golden_cross_count += 1
                elif prev_k >= prev_d and k_value < d_value:
                    death_cross = True
            momentum_5 = (prices[step] - prices[step - 5]) / prices[step - 5] if step >= 5 else 0
            momentum_10 = (prices[step] - prices[step - 10]) / prices[step - 10] if step >= 10 else 0
            volumes = current_indicators['VOLUMES']
            if step >= 5:
                avg_volume = np.mean(volumes[step - 5:step])
                current_volume = volumes[step] if step < len(volumes) else avg_volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_ratio = 1
            if position != 0 and entry_price > 0:
                atr = atr_values_14[step] if step < len(atr_values_14) else self.env.current_price * 0.02
                base_atr_mult = params.get('atr_multiplier', 1.8)
                if position > 0:
                    profit_pct = (self.env.current_price - entry_price) / entry_price
                    if golden_cross_count >= 2:
                        atr_mult = base_atr_mult * 1.2
                    elif profit_pct > 0.05:
                        atr_mult = base_atr_mult * 0.8
                    else:
                        atr_mult = base_atr_mult
                    stop_price = entry_price - atr_mult * atr
                    if profit_pct > 0.04:
                        trailing_stop = self.env.current_price * 0.97
                        stop_price = max(stop_price, trailing_stop)
                    should_close = False
                    close_reason = ''
                    if death_cross and k_value > 70:
                        should_close = True
                        close_reason = 'kdj_death_cross'
                    elif j_value > 80 and j_momentum < -10:
                        should_close = True
                        close_reason = 'j_reversion'
                    elif self.env.current_price < stop_price:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif profit_pct > 0.08 and (j_value > 70 or k_value > 70):
                        should_close = True
                        close_reason = 'profit_target'
                    if should_close:
                        self._close_long_position()
                        position = 0
                        position_layers = 0
                        golden_cross_count = 0
                        trades.append({'step': step, 'action': 'close', 'price': self.env.current_price, 'signal': f'{close_reason}_j{j_value:.0f}_k{k_value:.0f}'})
                        if step > start_step:
                            self._update_daily_equity()
                        continue
            if position == 0:
                buy_score = 0
                if j_value < j_buy_threshold:
                    buy_score += 2
                elif j_value < j_buy_threshold + 10:
                    buy_score += 1
                if k_value < k_buy_threshold:
                    buy_score += 1
                if d_value < 30:
                    buy_score += 1
                if golden_cross and k_value < 50:
                    buy_score += 2
                if j_value < 0:
                    buy_score += 1
                if momentum_5 < -0.02 and volume_ratio > 1.2:
                    buy_score += 1
                if j_momentum > 5 and j_value < 20:
                    buy_score += 1
                if buy_score >= 3:
                    if buy_score >= 6:
                        position_size = 0.9
                    elif buy_score >= 4:
                        position_size = 0.85
                    else:
                        position_size = 0.75
                    self._open_long_position(position_size)
                    position = 1
                    position_layers = 1
                    entry_price = self.env.current_price
                    entry_prices = [step]
                    trades.append({'step': step, 'action': 'buy', 'price': self.env.current_price, 'signal': f"kdj_oversold_j{j_value:.0f}_k{k_value:.0f}_s{buy_score}{('_gc' if golden_cross else '')}", 'shares': self.env.shares_held})
                elif j_value > j_sell_threshold and k_value > k_sell_threshold and (not is_short_window):
                    if death_cross or (j_momentum < -10 and volume_ratio > 1.3):
                        self._open_short_position(0.7)
                        position = -1
                        entry_price = self.env.current_price
                        entry_prices = [step]
                        trades.append({'step': step, 'action': 'sell', 'price': self.env.current_price, 'signal': f'kdj_overbought_j{j_value:.0f}_k{k_value:.0f}', 'shares': self.env.shares_held})
            elif position > 0 and position_layers < max_layers and allow_pyramid:
                add_signal = False
                add_reason = ''
                if j_value < j_buy_threshold - 10 * position_layers:
                    add_signal = True
                    add_reason = 'j_extreme'
                elif golden_cross and golden_cross_count > 1 and (k_value < 40):
                    add_signal = True
                    add_reason = 'golden_cross'
                elif self.env.current_price < entry_price * 0.98 and j_momentum > 0:
                    add_signal = True
                    add_reason = 'kdj_divergence'
                elif position_layers == 1 and step - trades[-1]['step'] > 3:
                    if j_value < 30 and k_value < 40:
                        add_signal = True
                        add_reason = 'time_add'
                if add_signal:
                    add_size = 0.2 + 0.05 * position_layers
                    self._add_to_position(add_size)
                    position_layers += 1
                    entry_prices.append(step)
                    trades.append({'step': step, 'action': 'add', 'price': self.env.current_price, 'signal': f'kdj_add_{add_reason}_layer{position_layers}', 'shares': self.env.shares_held})
            if step > start_step:
                self._update_daily_equity()
        final_step = end_step - 1
        if position > 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_long_position()
            trades.append({'step': final_step, 'action': 'final_close_long', 'price': self.env.current_price, 'signal': 'window_end'})
        elif position < 0:
            self.env.current_step = final_step
            self.env.current_price = self._price_at(final_step)
            self._close_short_position()
            trades.append({'step': final_step, 'action': 'final_close_short', 'price': self.env.current_price, 'signal': 'window_end'})
        self._update_daily_equity()
        final_portfolio_value = self.env.portfolio_value
        return self._calculate_strategy_metrics(trades, start_step, end_step, final_portfolio_value)

    def _update_daily_equity(self):
        if hasattr(self.env, 'short_shares') and self.env.short_shares > 0:
            equity = self.env.balance - self.env.short_shares * self.env.current_price
        else:
            equity = self.env.balance + self.env.shares_held * self.env.current_price
        self.env.portfolio_value = equity
        self.env.daily_equity_curve.append(equity)

    def _close_long_position(self):
        if self.env.shares_held > 0 and self.env.current_price > 0:
            shares_to_sell = self.env.shares_held
            proceeds = shares_to_sell * self.env.current_price * (1 - COMMISSION_RATE)
            self.env.balance += proceeds
            self.env.shares_held = 0
            self.env.portfolio_value = self.env.balance
            self.env.trades.append({'type': 'sell', 'step': self.env.current_step, 'price': self.env.current_price, 'shares': shares_to_sell, 'proceeds': proceeds, 'portfolio_value': self.env.portfolio_value})

    def _close_short_position(self):
        if hasattr(self.env, 'short_shares') and self.env.short_shares > 0 and (self.env.current_price > 0):
            shares_to_cover = self.env.short_shares
            cost = shares_to_cover * self.env.current_price * (1 + COMMISSION_RATE)
            profit = shares_to_cover * (self.env.short_entry_price - self.env.current_price) - shares_to_cover * (self.env.short_entry_price * COMMISSION_RATE + self.env.current_price * COMMISSION_RATE)
            self.env.balance -= cost
            self.env.short_shares = 0
            self.env.portfolio_value = self.env.balance
            self.env.trades.append({'type': 'cover', 'step': self.env.current_step, 'price': self.env.current_price, 'shares': shares_to_cover, 'cost': cost, 'profit': profit, 'portfolio_value': self.env.portfolio_value})

    def _calculate_atr(self, period=14):
        if period in self._cached_atr and self._cached_atr[period]:
            return self._cached_atr[period]
        try:
            hist_data = self.env.data['analysis_100days']['analysis_data']
            high_prices = np.asarray(hist_data['HIGH_PRICES'], dtype=np.float64)
            low_prices = np.asarray(hist_data['LOW_PRICES'], dtype=np.float64)
            close_prices = np.asarray(hist_data['CLOSING_PRICES'], dtype=np.float64)
        except Exception:
            return [0]
        result = self._compute_atr_vectorized(high_prices, low_prices, close_prices, period)
        self._cached_atr[period] = result
        return result

    def _calculate_strategy_metrics(self, trades, start_step, end_step, final_portfolio_value=None):
        if final_portfolio_value is not None:
            final_value = final_portfolio_value
        else:
            final_value = self.env.portfolio_value
        initial_value = self.env.initial_balance
        if initial_value > 0:
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0
        max_len = len(self.df) if self.df is not None else len(self.env.prices) if self.env.prices else 0
        if start_step < max_len and end_step <= max_len:
            try:
                start_price = self._price_at(start_step)
                end_price = self._price_at(min(end_step - 1, max_len - 1))
                buy_hold_return = (end_price - start_price) / start_price
            except Exception as e:
                print(f'Warning: Failed to calculate buy_hold_return: {e}')
                buy_hold_return = 0
        else:
            buy_hold_return = 0
        non_trading_actions = {'hold_cash', 'hold', 'init'}
        real_trades = [t for t in trades if t.get('action', '') not in non_trading_actions]
        equity_curve = self.env.daily_equity_curve
        if len(equity_curve) < 2:
            return {'total_return': total_return, 'buy_hold_return': buy_hold_return, 'excess_return': total_return - buy_hold_return, 'sharpe_ratio': 0, 'max_drawdown': 0, 'num_trades': len(real_trades), 'trades': trades, 'final_portfolio_value': final_value, 'equity_curve': equity_curve, 'avg_position_size': 0}
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_return = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                daily_returns.append(daily_return)
        if len(daily_returns) >= 2:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            epsilon = max(1e-06, abs(mean_return) * 0.01)
            std_return_safe = std_return + epsilon
            sharpe_ratio_raw = mean_return / std_return_safe * np.sqrt(252)
            sharpe_scale = 3.0
            sharpe_ratio = sharpe_scale * np.tanh(sharpe_ratio_raw / sharpe_scale)
        else:
            sharpe_ratio = 0
        if self.debug_mode and len(daily_returns) > 0:
            if np.allclose(daily_returns, 0, atol=1e-10):
                assert abs(sharpe_ratio) < 1e-06, f'daily_returns  0  sharpe_ratio={sharpe_ratio:.6f} != 0'
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        daily_position_sizes = []
        current_shares = 0
        short_shares = 0
        for day in range(start_step, end_step):
            day_trades = [t for t in trades if t['step'] == day]
            for trade in day_trades:
                action = trade['action']
                if action in ['buy', 'open_long']:
                    current_shares = trade.get('shares', 0)
                    short_shares = 0
                elif action in ['sell', 'close', 'close_long', 'take_profit', 'stop_loss', 'trailing_stop']:
                    current_shares = 0
                elif action in ['short', 'open_short']:
                    short_shares = trade.get('shares', 0)
                    current_shares = 0
                elif action in ['cover', 'close_short']:
                    short_shares = 0
                elif action == 'add':
                    current_shares += trade.get('shares', 0)
            if day < len(self.env.prices):
                price = self._price_at(day)
                if day - start_step < len(equity_curve):
                    total_value = equity_curve[day - start_step]
                    if total_value > 0 and price > 0:
                        position_value = (current_shares + short_shares) * price
                        position_size = position_value / total_value
                        daily_position_sizes.append(min(position_size, 1.0))
                    else:
                        daily_position_sizes.append(0)
                else:
                    daily_position_sizes.append(0)
            else:
                daily_position_sizes.append(0)
        if daily_position_sizes:
            avg_position_size = np.mean(daily_position_sizes)
        else:
            avg_position_size = 0
        if len(trades) > 0 and trades[0].get('signal', '').startswith('cash_only'):
            avg_position_size = 0
        win_trades = len([t for t in trades if t['action'] in ['close', 'take_profit', 'trailing_stop']])
        loss_trades = len([t for t in trades if t['action'] == 'stop_loss'])
        win_rate = win_trades / (win_trades + loss_trades) if win_trades + loss_trades > 0 else 0
        sortino_ratio = 0
        if len(daily_returns) > 1:
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino_ratio = mean_return / downside_std * np.sqrt(252)
        return {'total_return': total_return, 'buy_hold_return': buy_hold_return, 'excess_return': total_return - buy_hold_return, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio, 'max_drawdown': max_drawdown, 'num_trades': len(real_trades), 'trades': trades, 'final_portfolio_value': final_value, 'equity_curve': equity_curve, 'daily_returns': daily_returns, 'avg_position_size': avg_position_size, 'volatility': self.volatility, 'atr_ratio': self.atr_ratio, 'win_rate': win_rate, 'win_trades': win_trades, 'loss_trades': loss_trades, 'actual_trading_days': end_step - start_step}

    def _recompute_indicators(self, prices, high_prices=None, low_prices=None, volumes=None):
        if high_prices is None:
            high_prices = [p * 1.01 for p in prices]
        if low_prices is None:
            low_prices = [p * 0.99 for p in prices]
        if volumes is None:
            volumes = [1000000] * len(prices)
        indicators = {'CLOSING_PRICES': prices, 'HIGH_PRICES': high_prices, 'LOW_PRICES': low_prices, 'VOLUMES': volumes}
        indicators['MA10_VALUES'] = self._calculate_ma_series(prices, 10)
        indicators['MA20_VALUES'] = self._calculate_ma_series(prices, 20)
        indicators['MA100_VALUES'] = self._calculate_ma_series(prices, 100)
        indicators['RSI_VALUES'] = self._calculate_rsi_series(prices, 14)
        upper, middle, lower = self._calculate_bollinger_bands_series(prices, 20, 2)
        indicators['BOLL_UPPER_VALUES'] = upper
        indicators['BOLL_MIDDLE_VALUES'] = middle
        indicators['BOLL_LOWER_VALUES'] = lower
        k, d, j = self._calculate_kdj_series(high_prices, low_prices, prices)
        indicators['KDJ_K_VALUES'] = k
        indicators['KDJ_D_VALUES'] = d
        indicators['KDJ_J_VALUES'] = j
        return indicators

    def _calculate_ma_series(self, prices, period):
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return []
        result = arr.copy()
        if n >= period:
            cs = np.cumsum(arr)
            result[period - 1] = cs[period - 1] / period
            if n > period:
                result[period:] = (cs[period:] - cs[:n - period]) / period
        return result.tolist()

    def _calculate_rsi_series(self, prices, period=14):
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period if len(seed[seed >= 0]) > 0 else 0
        down = -seed[seed < 0].sum() / period if len(seed[seed < 0]) > 0 else 0
        if down == 0:
            rs = 100 if up > 0 else 1
        else:
            rs = up / down
        rsi = [50.0] * period
        rsi.append(100 - 100 / (1 + rs))
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            if down == 0:
                rs = 100 if up > 0 else 1
            else:
                rs = up / down
            rsi.append(100 - 100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands_series(self, prices, window=20, num_std=2):
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        middle_arr = np.asarray(self._calculate_ma_series(arr, window), dtype=np.float64)
        upper_arr = arr.copy()
        lower_arr = arr.copy()
        if n >= window:
            shape = (n - window + 1, window)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            rolling_std = windows.std(axis=1, ddof=0)
            upper_arr[window - 1:] = middle_arr[window - 1:] + num_std * rolling_std
            lower_arr[window - 1:] = middle_arr[window - 1:] - num_std * rolling_std
        return (upper_arr.tolist(), middle_arr.tolist(), lower_arr.tolist())

    def _calculate_kdj_series(self, high_prices, low_prices, close_prices, n=9):
        high = np.asarray(high_prices, dtype=np.float64)
        low = np.asarray(low_prices, dtype=np.float64)
        close = np.asarray(close_prices, dtype=np.float64)
        length = len(close)
        k_values = np.full(length, 50.0)
        d_values = np.full(length, 50.0)
        j_values = np.full(length, 50.0)
        if length >= n:
            shape = (length - n + 1, n)
            strides = (high.strides[0], high.strides[0])
            high_windows = np.lib.stride_tricks.as_strided(high, shape=shape, strides=strides)
            low_windows = np.lib.stride_tricks.as_strided(low, shape=shape, strides=strides)
            period_high = high_windows.max(axis=1)
            period_low = low_windows.min(axis=1)
            denom = period_high - period_low
            rsv = np.where(denom > 0, (close[n - 1:] - period_low) / denom * 100, 50.0)
            k, d = (rsv[0], rsv[0])
            k_values[n - 1] = k
            d_values[n - 1] = d
            j_values[n - 1] = 3 * k - 2 * d
            for i in range(1, len(rsv)):
                k = 2.0 / 3.0 * k + 1.0 / 3.0 * rsv[i]
                d = 2.0 / 3.0 * d + 1.0 / 3.0 * k
                idx = n - 1 + i
                k_values[idx] = k
                d_values[idx] = d
                j_values[idx] = 3 * k - 2 * d
        return (k_values.tolist(), d_values.tolist(), j_values.tolist())

    def _get_current_indicators(self, start_step, current_step):
        if self._cached_full_indicators is not None:
            return self._cached_full_indicators
        lookback_needed = 100
        hist_prices = self._hist_close_arr
        hist_high = self._hist_high_arr
        hist_low = self._hist_low_arr
        hist_volumes = self._hist_volume_arr
        end = max(0, current_step + 1)
        if self._close_prices_arr is None:
            current_prices = np.asarray([self._price_at(s) for s in range(end)], dtype=np.float64)
        else:
            current_prices = self._close_prices_arr[:end]
        current_high = self._high_prices_arr[:end] if self._high_prices_arr is not None else current_prices * 1.01
        current_low = self._low_prices_arr[:end] if self._low_prices_arr is not None else current_prices * 0.99
        current_volumes = np.full(end, 1000000.0, dtype=np.float64)
        if start_step > 0:
            combined_prices = np.concatenate([hist_prices, current_prices])
            combined_high = np.concatenate([hist_high, current_high])
            combined_low = np.concatenate([hist_low, current_low])
            combined_volumes = np.concatenate([hist_volumes, current_volumes])
        else:
            combined_prices = current_prices
            combined_high = current_high
            combined_low = current_low
            combined_volumes = current_volumes
        if len(combined_prices) < lookback_needed:
            padding_needed = lookback_needed - len(combined_prices)
            combined_prices = np.pad(combined_prices, (padding_needed, 0), mode='edge')
            combined_high = np.pad(combined_high, (padding_needed, 0), mode='edge')
            combined_low = np.pad(combined_low, (padding_needed, 0), mode='edge')
            combined_volumes = np.pad(combined_volumes, (padding_needed, 0), mode='edge')
        return self._recompute_indicators(combined_prices, combined_high, combined_low, combined_volumes)

    def _calculate_bollinger_bands_adaptive(self, prices, window, num_std):
        arr = np.asarray(prices, dtype=np.float64)
        n = len(arr)
        middle_arr = np.asarray(self._calculate_ma_series(arr, window), dtype=np.float64)
        upper_arr = arr.copy()
        lower_arr = arr.copy()
        if n >= window:
            shape = (n - window + 1, window)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            rolling_std = windows.std(axis=1, ddof=0)
            upper_arr[window - 1:] = middle_arr[window - 1:] + num_std * rolling_std
            lower_arr[window - 1:] = middle_arr[window - 1:] - num_std * rolling_std
        return (upper_arr.tolist(), middle_arr.tolist(), lower_arr.tolist())

class TradingEnvironment:

    def __init__(self, data, image_path=None, processor=None, future_prices=None, df=None, close_col='close', high_col='high', low_col='low', volume_col='vol'):
        self.data = data
        self.future_prices = future_prices
        self.prices = None
        self.current_step = 0
        self.df = df
        self.close_col = close_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col
        self.initial_balance = INITIAL_BALANCE
        self.balance = self.initial_balance
        self.shares_held = 0
        self.short_shares = 0
        self.short_entry_price = 0
        self.portfolio_value = self.initial_balance
        self.returns = []
        self.trades = []
        self.daily_equity_curve = []
        self.window_size = WINDOW_SIZE
        self.image_path = image_path
        self.processor = processor
        self.image_features = None
        self.multiscale_images = (None, None)
        self.timeseries_data = None
        self.commission_rate = COMMISSION_RATE
        self.buy_hold_initial_price = None
        self.buy_hold_shares = 0
        self.strategy_executor = AdaptiveStrategyExecutor(self)
        self.current_strategy = None
        self.strategy_start_step = 0
        self.expert_portfolios = {}
        self.expert_strategies = {}
        self.expert_daily_returns = {}
        self.debug_mode = False
        if image_path and os.path.exists(image_path) and processor:
            self._load_image_features()
            self._load_multiscale_images()
        else:
            self.multiscale_images = (None, None)
        self.reset()

    def get_price(self, step, col='close'):
        price = 0.0
        if self.df is not None and step < len(self.df):
            if col == 'close' and self.close_col in self.df.columns:
                price = float(self.df.loc[step, self.close_col])
            elif col == 'high' and self.high_col in self.df.columns:
                price = float(self.df.loc[step, self.high_col])
            elif col == 'low' and self.low_col in self.df.columns:
                price = float(self.df.loc[step, self.low_col])
            else:
                price = float(self.df.loc[step, self.close_col]) if self.close_col in self.df.columns else 0.0
        elif self.prices is not None and step < len(self.prices):
            price = float(self.prices[step])
        else:
            price = 0.0
        if not np.isfinite(price):
            raise ValueError(f'Price at step {step} is not finite: {price}')
        if price <= 0:
            raise ValueError(f'Price at step {step} is non-positive: {price}')
        return price

    def calculate_buy_and_hold(self, start_step=0, end_step=None):
        if end_step is None:
            end_step = len(self.prices)
        initial_price = self.prices[start_step] if start_step < len(self.prices) else 0
        final_price = self.prices[min(end_step - 1, len(self.prices) - 1)] if end_step <= len(self.prices) else 0
        if initial_price <= 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'daily_returns': []}
        buy_hold_position = BUY_HOLD_CONFIG.get('position_size', 0.85)
        available_cash = self.initial_balance * buy_hold_position
        buy_hold_shares = int(available_cash / (initial_price * (1 + COMMISSION_RATE)))
        buy_cost = buy_hold_shares * initial_price * (1 + COMMISSION_RATE)
        sell_proceeds = buy_hold_shares * final_price * (1 - COMMISSION_RATE)
        remaining_cash = self.initial_balance - buy_cost
        final_value = sell_proceeds + remaining_cash
        total_return = (final_value - self.initial_balance) / self.initial_balance
        daily_returns = []
        for i in range(start_step + 1, min(end_step, len(self.prices))):
            if self.prices[i - 1] > 0:
                stock_return = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
                daily_return = buy_hold_position * stock_return
                daily_returns.append(daily_return)
        if len(daily_returns) >= 2:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            if std_return >= 1e-08:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        peak = initial_price
        max_drawdown = 0
        for i in range(start_step, min(end_step, len(self.prices))):
            if self.prices[i] > peak:
                peak = self.prices[i]
            drawdown = (peak - self.prices[i]) / peak * buy_hold_position if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        return {'total_return': total_return, 'sharpe_ratio': sharpe_ratio, 'max_drawdown': max_drawdown, 'daily_returns': daily_returns, 'final_value': final_value, 'position_size': buy_hold_position}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.short_shares = 0
        self.short_entry_price = 0
        self.portfolio_value = self.initial_balance
        self.returns = []
        self.trades = []
        self.daily_equity_curve = []
        self.current_strategy = None
        self.strategy_start_step = 0
        self.expert_portfolios = {}
        self.expert_strategies = {}
        self.expert_daily_returns = {}
        if self.future_prices is not None:
            self.prices = self.future_prices
            self.current_price = self.prices[0] if len(self.prices) > 0 else 0
        elif self.df is not None:
            self.prices = self.df[self.close_col].values
            self.current_price = self.prices[0] if len(self.prices) > 0 else 0
        else:
            analysis_data = self.data['analysis_100days']['analysis_data']
            self.prices = analysis_data['CLOSING_PRICES']
            self.current_price = self.prices[-1] if len(self.prices) > 0 else 0
        if len(self.prices) > 0:
            self.buy_hold_initial_price = self.prices[0]
            buy_hold_position = BUY_HOLD_CONFIG.get('position_size', 0.85)
            available_cash = self.initial_balance * buy_hold_position
            self.buy_hold_shares = int(available_cash / (self.buy_hold_initial_price * (1 + COMMISSION_RATE)))
        if len(self.prices) < self.window_size:
            print(f'warn({len(self.prices)})less than ({self.window_size})')
            self.window_size = len(self.prices)
        self._extract_timeseries_features()
        return self._get_state()

    def _load_image_features(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            if self.processor:
                processed = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                self.image_features = processed
            else:
                self.image_features = None
        except Exception as e:
            print(f'loading figures error: {e}')
            self.image_features = None

    def _load_multiscale_images(self):
        try:
            from data import extract_multiscale_images
            global_tensor, local_tensor = extract_multiscale_images(self.image_path, self.processor)
            self.multiscale_images = (global_tensor, local_tensor)
        except Exception as e:
            print(f'error: {e}')
            self.multiscale_images = (None, None)

    def _extract_timeseries_features(self):
        try:
            from data import extract_timeseries_data
            state = self._get_state()
            self.timeseries_data = extract_timeseries_data(state)
        except Exception as e:
            print(f'error: {e}')
            self.timeseries_data = None

    def _get_state(self):
        return {'current_step': self.current_step, 'balance': self.balance, 'shares_held': self.shares_held, 'portfolio_value': self.portfolio_value, 'current_price': self.current_price, 'data': self.data, 'image_features': self.image_features, 'multiscale_images': self.multiscale_images, 'timeseries_data': self.timeseries_data, 'current_strategy': self.current_strategy, 'expert_portfolios': self.expert_portfolios}

    def step(self, action):
        if self.current_strategy != action:
            self.current_strategy = action
            self.strategy_start_step = self.current_step
            end_step = min(self.current_step + self.window_size, len(self.prices))
            try:
                strategy_result = self.strategy_executor.execute_strategy(action, self.current_step, end_step)
            except Exception as e:
                print(f'Strategy execution error: {e}')
                strategy_result = {'total_return': 0, 'buy_hold_return': 0, 'excess_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'num_trades': 0}
            buy_hold_result = self.calculate_buy_and_hold(self.current_step, end_step)
            self.current_step = end_step - 1
            if self.current_step >= len(self.prices) - 1:
                done = True
            else:
                done = False
            self.portfolio_value = self.balance + self.shares_held * self.current_price
            if hasattr(self, 'short_shares') and self.short_shares > 0:
                short_pnl = self.short_shares * (self.short_entry_price - self.current_price)
                self.portfolio_value = self.balance + short_pnl
            reward = strategy_result['total_return']
            next_state = self._get_state()
            info = {'step_return': strategy_result['total_return'], 'buy_hold_return': buy_hold_result['total_return'], 'excess_return': strategy_result.get('excess_return', strategy_result['total_return'] - buy_hold_result['total_return']), 'sharpe_ratio': strategy_result['sharpe_ratio'], 'portfolio_value': self.portfolio_value, 'max_drawdown': strategy_result['max_drawdown'], 'num_trades': strategy_result['num_trades'], 'strategy': STRATEGY_ACTIONS[action]['name'], 'action': action, 'adaptive_params': strategy_result.get('adaptive_params', {}), 'avg_position_size': strategy_result.get('avg_position_size', 0.8), 'win_rate': strategy_result.get('win_rate', 0)}
            return (next_state, reward, done, info)
        else:
            self.current_step += 1
            if self.current_step >= len(self.prices):
                return (self._get_state(), 0, True, {})
            self.current_price = self.prices[self.current_step]
            if hasattr(self, 'short_shares') and self.short_shares > 0:
                short_pnl = self.short_shares * (self.short_entry_price - self.current_price)
                self.portfolio_value = self.balance + short_pnl
            else:
                self.portfolio_value = self.balance + self.shares_held * self.current_price
            if self.current_step > 0 and self.initial_balance > 0:
                step_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
            else:
                step_return = 0
            buy_hold_return = 0
            if self.buy_hold_initial_price > 0:
                buy_hold_return = (self.current_price - self.buy_hold_initial_price) / self.buy_hold_initial_price
            done = self.current_step >= len(self.prices) - 1
            next_state = self._get_state()
            info = {'step_return': step_return, 'buy_hold_return': buy_hold_return, 'excess_return': step_return - buy_hold_return, 'sharpe_ratio': 0, 'portfolio_value': self.portfolio_value, 'action': action}
            return (next_state, step_return, done, info)

    def step_multi_expert(self, expert_weights, expert_actions):
        if len(expert_weights) != NUM_EXPERTS or len(expert_actions) != NUM_EXPERTS:
            raise ValueError(f'The number of expert weights must equal the number of actions {NUM_EXPERTS}')
        portfolio_value_at_start = 0
        if not self.expert_portfolios:
            print(f'\n===  {NUM_EXPERTS}  ===')
            for i in range(NUM_EXPERTS):
                allocated_capital = self.initial_balance * expert_weights[i]
                self.expert_portfolios[i] = {'balance': allocated_capital, 'initial_balance': allocated_capital, 'shares': 0, 'short_shares': 0, 'portfolio_value': allocated_capital, 'trades': [], 'daily_equity_curve': [], 'last_allocation': allocated_capital}
                self.expert_strategies[i] = expert_actions[i]
                self.expert_daily_returns[i] = []
                if self.debug_mode:
                    print(f'expert {i}: allocation ${allocated_capital:.2f} ({expert_weights[i]:.1%})')
                portfolio_value_at_start += allocated_capital
        else:
            portfolio_value_at_start = sum((self.expert_portfolios[i]['portfolio_value'] for i in range(NUM_EXPERTS)))
        if self.debug_mode:
            print(f'\nThe total portfolio value at the beginning of the episode: ${portfolio_value_at_start:.2f}')
        end_step = min(self.current_step + self.window_size, len(self.prices))
        current_round_buy_hold_return = self._calculate_current_round_buy_hold(self.current_step, end_step)
        expert_results = {}
        rewards_dict = {}
        print(f'\n=== excute {NUM_EXPERTS} strategies ===')
        for i in range(NUM_EXPERTS):
            if expert_weights[i] <= 0:
                expert_results[i] = {'total_return': 0, 'sharpe_ratio': 0, 'daily_returns': [], 'final_portfolio_value': self.expert_portfolios[i]['initial_balance'], 'num_trades': 0}
                rewards_dict[i] = (0, 0)
                self.expert_daily_returns[i] = []
                continue
            current_capital = self.expert_portfolios[i]['balance']
            if self.debug_mode:
                print(f"\nexpert {i} ({STRATEGY_ACTIONS[expert_actions[i]]['name']}):")
                print(f'  weights: {expert_weights[i]:.4f} ({expert_weights[i] * 100:.2f}%)')
                print(f'  captical: ${current_capital:.2f}')
                print(f'  strategyID: {expert_actions[i]}')
            try:
                result = self.strategy_executor.execute_strategy_isolated(expert_actions[i], self.current_step, end_step, current_capital)
                if self.debug_mode:
                    print(f'  Execution Results:')
                    print(f"    Total Return: {result['total_return']:.4f} ({result['total_return'] * 100:.2f}%)")
                    print(f"    Number of Trades: {result['num_trades']}")
                    print(f"    Final Portfolio Value: ${result['final_portfolio_value']:.2f}")
                    print(f"    Sharpe Ratio: {result['sharpe_ratio']:.3f}")
                if result['total_return'] == 0 and result['num_trades'] > 0:
                    print(f"Warning: Expert {i} executed {result['num_trades']} trades but achieved zero return!")
                self.expert_portfolios[i]['balance'] = result['final_balance']
                self.expert_portfolios[i]['shares'] = result.get('final_shares', 0)
                self.expert_portfolios[i]['portfolio_value'] = result['final_portfolio_value']
                self.expert_portfolios[i]['trades'].extend(result.get('trades', []))
                if 'equity_curve' in result:
                    self.expert_portfolios[i]['daily_equity_curve'] = result['equity_curve']
                if 'daily_returns' in result:
                    self.expert_daily_returns[i] = result['daily_returns']
                expert_results[i] = result
                rewards_dict[i] = (result['total_return'], result['sharpe_ratio'])
            except Exception as e:
                print(f'Error occurred while Expert {i} was executing strategy {expert_actions[i]}: {e}')
                expert_results[i] = {'total_return': 0, 'sharpe_ratio': 0, 'daily_returns': [], 'final_portfolio_value': current_capital, 'num_trades': 0, 'error': str(e)}
                rewards_dict[i] = (0, 0)
                self.expert_daily_returns[i] = []
        self.current_step = end_step - 1
        done = self.current_step >= len(self.prices) - 1
        total_portfolio_value_at_end = sum((self.expert_portfolios[i]['portfolio_value'] for i in range(NUM_EXPERTS)))
        self.portfolio_value = total_portfolio_value_at_end
        if portfolio_value_at_start > 0:
            current_round_return = (total_portfolio_value_at_end - portfolio_value_at_start) / portfolio_value_at_start
        else:
            current_round_return = 0
        if self.debug_mode:
            print(f'Total portfolio value at the end of the episode: ${total_portfolio_value_at_end:.2f}')
            print(f'Return for the current episode: {current_round_return:.4f} ({current_round_return * 100:.2f}%)')
            print(f'Change in portfolio value: ${total_portfolio_value_at_end - portfolio_value_at_start:.2f}')
        combined_daily_returns = []
        max_length = max((len(self.expert_daily_returns.get(i, [])) for i in range(NUM_EXPERTS)))
        if max_length > 0:
            for day in range(max_length):
                daily_combined_return = 0
                for i in range(NUM_EXPERTS):
                    if i in self.expert_daily_returns and day < len(self.expert_daily_returns[i]):
                        daily_combined_return += expert_weights[i] * self.expert_daily_returns[i][day]
                combined_daily_returns.append(daily_combined_return)
            if len(combined_daily_returns) > 1:
                mean_return = np.mean(combined_daily_returns)
                std_return = np.std(combined_daily_returns, ddof=1)
                if std_return > 0:
                    combined_sharpe = mean_return / std_return * np.sqrt(252)
                else:
                    combined_sharpe = 0
            else:
                combined_sharpe = 0
        else:
            combined_sharpe = 0
        print(f'\n=== Summary of Execution Results ===')
        print(f'Return for the current episode: {current_round_return:.4f} ({current_round_return * 100:.2f}%)')
        print(f'Buy-Hold Return: {current_round_buy_hold_return:.4f} ({current_round_buy_hold_return * 100:.2f}%)')
        print(f'Excess Return: {current_round_return - current_round_buy_hold_return:.4f}')
        print(f'Combined Sharpe Ratio: {combined_sharpe:.3f}')
        print(f'Portfolio Value at Episode Start: ${portfolio_value_at_start:.2f}')
        print(f'Portfolio Value at Episode End: ${total_portfolio_value_at_end:.2f}')
        print(f'Change in Portfolio Value: ${total_portfolio_value_at_end - portfolio_value_at_start:.2f}')
        for i in range(NUM_EXPERTS):
            result = expert_results[i]
            print(f"expert{i}: {result['total_return'] * 100:.2f}% (${self.expert_portfolios[i]['portfolio_value']:.2f})")
        next_state = self._get_state()
        info = {'total_return': current_round_return, 'buy_hold_return': current_round_buy_hold_return, 'excess_return': current_round_return - current_round_buy_hold_return, 'weighted_sharpe': combined_sharpe, 'portfolio_value': total_portfolio_value_at_end, 'portfolio_value_start': portfolio_value_at_start, 'portfolio_value_end': total_portfolio_value_at_end, 'portfolio_value_change': total_portfolio_value_at_end - portfolio_value_at_start, 'expert_results': expert_results, 'expert_weights': expert_weights, 'expert_actions': expert_actions, 'expert_portfolios': self.expert_portfolios, 'combined_daily_returns': combined_daily_returns, 'adaptive_params': {i: result.get('adaptive_params', {}) for i, result in expert_results.items()}, 'market_condition': self._analyze_market_condition(), 'cumulative_return': (total_portfolio_value_at_end - self.initial_balance) / self.initial_balance, 'cumulative_portfolio_value': total_portfolio_value_at_end, 'initial_balance': self.initial_balance}
        return (next_state, rewards_dict, done, info)

    def _calculate_current_round_buy_hold(self, start_step, end_step):
        if start_step >= len(self.prices) or end_step > len(self.prices):
            return 0
        start_price = self.prices[start_step]
        end_price = self.prices[min(end_step - 1, len(self.prices) - 1)]
        if start_price <= 0:
            return 0
        buy_hold_position = BUY_HOLD_CONFIG.get('position_size', 0.85)
        price_return = (end_price - start_price) / start_price
        total_commission = COMMISSION_RATE * 2
        buy_hold_return = buy_hold_position * (price_return - total_commission)
        return buy_hold_return

    def _analyze_market_condition(self):
        cur = self.current_step
        lookback = 20
        if cur < lookback or len(self.prices) <= cur:
            return {'trend': 'neutral', 'volatility': 'normal'}
        price_now = self.prices[cur]
        price_prev = self.prices[cur - lookback]
        recent_return = (price_now - price_prev) / price_prev if price_prev > 0 else 0
        if recent_return > 0.05:
            trend = 'bullish'
        elif recent_return < -0.05:
            trend = 'bearish'
        else:
            trend = 'neutral'
        returns = []
        for i in range(cur - lookback + 1, cur + 1):
            if self.prices[i - 1] > 0:
                ret = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
                returns.append(ret)
        if returns:
            volatility = np.std(returns) * np.sqrt(252)
            if volatility > 0.3:
                vol_level = 'high'
            elif volatility < 0.15:
                vol_level = 'low'
            else:
                vol_level = 'normal'
        else:
            vol_level = 'normal'
        return {'trend': trend, 'volatility': vol_level, 'recent_return': recent_return}
