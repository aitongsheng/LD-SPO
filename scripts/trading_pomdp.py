import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from ldspo_config import STRATEGY_ACTIONS, COMMISSION_RATE, INITIAL_BALANCE, NUMERICAL_FEATURE_COLS
from multimodal_observation import build_news_observation

class RLTradingEnvWrapper:

    def __init__(self, df: pd.DataFrame, window_size: int=30, lookback: int=60, start_idx: int=0, end_idx: Optional[int]=None, reward_config: Optional[Dict]=None, sequential_eval: bool=False, allow_short: bool=True):
        self.df = df.reset_index(drop=True)
        self.allow_short = allow_short
        self.window_size = window_size
        self.lookback = lookback
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.df)
        self.sequential_eval = sequential_eval
        self.reward_config = reward_config or {'return_weight': 0.7, 'sharpe_weight': 0.3, 'reward_scale_factor': 1.0}
        self.feature_cols = [c for c in NUMERICAL_FEATURE_COLS if c in self.df.columns]
        self._price_cols = {'open', 'high', 'low', 'close', 'ma_5', 'ma_10', 'ma_20', 'ma_60', 'boll_up', 'boll_down', 'change', 'atr_14'}
        self._volume_cols = {'vol', 'amount', 'vol_20d'}
        self._build_feature_matrix()
        self._mock_env = self._create_mock_env()
        from strategy_library import AdaptiveStrategyExecutor
        self.strategy_executor = AdaptiveStrategyExecutor(self._mock_env)
        self.strategy_executor.df = self.df
        self.current_step = self.lookback
        self.current_capital = INITIAL_BALANCE
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_returns = []
        self.episode_sharpes = []
        self.episode_trades = []
        self.episode_infos = []
        self.episode_portfolio_values = []
        self.episode_daily_equity = []

    def _build_feature_matrix(self):
        raw = self.df[self.feature_cols].copy().astype(np.float32)
        if 'close' in self.df.columns:
            close_vals = self.df['close'].values.astype(np.float32)
            close_vals = np.where(close_vals > 0, close_vals, 1.0)
            for col in self.feature_cols:
                if col in self._price_cols and col != 'close':
                    raw[col] = raw[col].values / close_vals
            raw['close'] = 1.0
        for col in self.feature_cols:
            if col in self._volume_cols:
                rolling_mean = self.df[col].rolling(window=20, min_periods=1).mean().values.astype(np.float32)
                rolling_mean = np.where(rolling_mean > 0, rolling_mean, 1.0)
                raw[col] = raw[col].values / rolling_mean
        self.feature_matrix = raw.fillna(0.0).values
        self.num_features = self.feature_matrix.shape[1]

    def _create_mock_env(self):

        class MockEnv:
            pass
        env = MockEnv()
        env.df = self.df
        env.close_col = 'close'
        env.high_col = 'high'
        env.low_col = 'low'
        env.volume_col = 'vol'
        env.allow_short = self.allow_short
        env.balance = INITIAL_BALANCE
        env.initial_balance = INITIAL_BALANCE
        env.shares_held = 0
        env.short_shares = 0
        env.short_entry_price = 0
        env.portfolio_value = INITIAL_BALANCE
        env.trades = []
        env.daily_equity_curve = []
        env.commission_rate = COMMISSION_RATE
        env.window_size = self.window_size
        env.prices = self.df['close'].values.tolist()
        env.current_price = env.prices[0] if len(env.prices) > 0 else 0
        env.current_step = 0
        env.data = {'analysis_100days': {'analysis_data': {'CLOSING_PRICES': self.df['close'].values.tolist(), 'HIGH_PRICES': self.df['high'].values.tolist() if 'high' in self.df.columns else self.df['close'].values.tolist(), 'LOW_PRICES': self.df['low'].values.tolist() if 'low' in self.df.columns else self.df['close'].values.tolist(), 'VOLUMES': self.df['vol'].values.tolist() if 'vol' in self.df.columns else [0] * len(self.df), 'RSI_VALUES': []}}}
        return env

    def _reset_mock_env(self):
        self._mock_env.balance = INITIAL_BALANCE
        self._mock_env.initial_balance = INITIAL_BALANCE
        self._mock_env.shares_held = 0
        self._mock_env.short_shares = 0
        self._mock_env.short_entry_price = 0
        self._mock_env.portfolio_value = INITIAL_BALANCE
        self._mock_env.trades = []
        self._mock_env.daily_equity_curve = []

    def reset(self, max_steps: Optional[int]=None) -> Dict[str, np.ndarray]:
        min_t = self.start_idx + self.lookback - 1
        if self.sequential_eval:
            self.current_step = min_t
            self.current_capital = INITIAL_BALANCE
        else:
            if max_steps is not None:
                max_t = self.end_idx - max_steps * self.window_size - 1
            else:
                max_t = self.end_idx - self.window_size - 1
            max_t = max(min_t, max_t)
            self.current_step = np.random.randint(min_t, max_t + 1)
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_returns = []
        self.episode_sharpes = []
        self.episode_trades = []
        self.episode_infos = []
        self.episode_portfolio_values = []
        self.episode_daily_equity = []
        self._reset_mock_env()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        t = min(self.current_step, self.end_idx - 1)
        start = max(0, t - self.lookback + 1)
        obs = self.feature_matrix[start:t + 1]
        if obs.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - obs.shape[0], self.num_features), dtype=np.float32)
            obs = np.concatenate([pad, obs], axis=0)
        news_token_ids, news_mask = build_news_observation(self.df, t)
        return {'numerical': obs, 'news_token_ids': news_token_ids, 'news_mask': news_mask}
    _TECHNICAL_STRATEGY_IDS = frozenset(range(8))
    _FALLBACK_THRESHOLD = 0.002

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        t = self.current_step
        exec_start = t + 1
        exec_end = min(exec_start + self.window_size, self.end_idx)
        initial_capital = self.current_capital if self.sequential_eval else INITIAL_BALANCE
        self._reset_mock_env()
        try:
            result = self.strategy_executor.execute_strategy_isolated(action, exec_start, exec_end, initial_capital)
        except Exception as e:
            result = {'total_return': 0.0, 'sharpe_ratio': 0.0, 'num_trades': 0, 'final_portfolio_value': initial_capital}
        fallback_triggered = False
        fallback_action = action
        fallback_direction = None
        fallback_price_change = 0.0
        original_strategy = STRATEGY_ACTIONS[action]['name']
        if action in self._TECHNICAL_STRATEGY_IDS and result.get('num_trades', 0) == 0:
            fallback_price_change = self._compute_window_price_change(exec_start, exec_end)
            if fallback_price_change > self._FALLBACK_THRESHOLD:
                fallback_action = 8
                fallback_direction = 'long'
            elif fallback_price_change < -self._FALLBACK_THRESHOLD:
                if self.allow_short:
                    fallback_action = 9
                    fallback_direction = 'short'
                else:
                    fallback_action = 10
                    fallback_direction = 'cash'
            if fallback_action != action:
                try:
                    fallback_result = self.strategy_executor.execute_strategy_isolated(fallback_action, exec_start, exec_end, initial_capital)
                    result = fallback_result
                    fallback_triggered = True
                except Exception:
                    pass
        total_return = result.get('total_return', 0.0)
        sharpe_ratio = result.get('sharpe_ratio', 0.0)
        num_trades = result.get('num_trades', 0)
        final_value = result.get('final_portfolio_value', initial_capital)
        daily_eq = result.get('debug_equity_curve', [])
        if daily_eq and len(daily_eq) > 0:
            if not self.episode_daily_equity:
                self.episode_daily_equity.append(float(initial_capital))
            self.episode_daily_equity.extend([float(v) for v in daily_eq])
        else:
            if not self.episode_daily_equity:
                self.episode_daily_equity.append(float(initial_capital))
            self.episode_daily_equity.append(float(final_value))
        reward = self._compute_reward(total_return, sharpe_ratio)
        if self.sequential_eval:
            self.current_capital = final_value
            self.current_step += self.window_size
            done = self.current_step + 1 + self.window_size > self.end_idx
        else:
            self.current_step += self.window_size
            done = self.current_step + 1 + self.window_size > self.end_idx
        info = {'total_return': total_return, 'sharpe_ratio': sharpe_ratio, 'num_trades': num_trades, 'final_value': final_value, 'portfolio_value': final_value, 'action': action, 'strategy': original_strategy, 'fallback_triggered': fallback_triggered, 'fallback_direction': fallback_direction, 'fallback_price_change': fallback_price_change, 'fallback_action': fallback_action, 'fallback_strategy': STRATEGY_ACTIONS[fallback_action]['name'] if fallback_triggered else original_strategy, 'original_strategy': original_strategy}
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_returns.append(total_return)
        self.episode_sharpes.append(sharpe_ratio)
        self.episode_trades.append(num_trades)
        self.episode_infos.append(info)
        self.episode_portfolio_values.append(final_value)
        obs_next = self._get_observation()
        return (obs_next, reward, done, info)

    def step_fusion(self, actions: np.ndarray, weights: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(actions) == 0 or len(actions) != len(weights):
            raise ValueError('actions and weights must be non-empty and equally sized')
        if np.any(actions < 0) or np.any(actions >= len(STRATEGY_ACTIONS)):
            raise ValueError('fusion action outside the strategy space')
        weights = np.maximum(weights, 0.0)
        weights = weights / max(weights.sum(), 1e-12)
        t = self.current_step
        exec_start = t + 1
        exec_end = min(exec_start + self.window_size, self.end_idx)
        initial_capital = self.current_capital if self.sequential_eval else INITIAL_BALANCE
        results = []
        for action in actions:
            try:
                result = self.strategy_executor.execute_strategy_isolated(int(action), exec_start, exec_end, initial_capital)
            except Exception:
                result = {'total_return': 0.0, 'sharpe_ratio': 0.0, 'num_trades': 0, 'final_portfolio_value': initial_capital, 'debug_equity_curve': []}
            results.append(result)
        strategy_returns = np.array([float(r.get('total_return', 0.0)) for r in results], dtype=np.float64)
        strategy_sharpes = np.array([float(r.get('sharpe_ratio', 0.0)) for r in results], dtype=np.float64)
        total_return = float(np.dot(weights, strategy_returns))
        sharpe_ratio = float(np.dot(weights, strategy_sharpes))
        final_value = float(initial_capital * (1.0 + total_return))
        num_trades = int(sum((int(r.get('num_trades', 0)) for r in results)))
        reward = self._compute_reward(total_return, sharpe_ratio)
        curves = [r.get('debug_equity_curve', []) for r in results]
        usable_lengths = [len(c) for c in curves if len(c) > 0]
        if usable_lengths:
            curve_len = min(usable_lengths)
            fused_curve = np.zeros(curve_len, dtype=np.float64)
            for weight, curve in zip(weights, curves):
                if len(curve) >= curve_len:
                    fused_curve += weight * np.asarray(curve[:curve_len], dtype=np.float64)
                else:
                    fused_curve += weight * initial_capital
            if not self.episode_daily_equity:
                self.episode_daily_equity.append(float(initial_capital))
            self.episode_daily_equity.extend(fused_curve.tolist())
        else:
            if not self.episode_daily_equity:
                self.episode_daily_equity.append(float(initial_capital))
            self.episode_daily_equity.append(final_value)
        if self.sequential_eval:
            self.current_capital = final_value
        self.current_step += self.window_size
        done = self.current_step + 1 + self.window_size > self.end_idx
        dominant_action = int(actions[int(np.argmax(weights))])
        allocation = {STRATEGY_ACTIONS[int(action)]['name']: float(weight) for action, weight in zip(actions, weights)}
        info = {'total_return': total_return, 'sharpe_ratio': sharpe_ratio, 'num_trades': num_trades, 'final_value': final_value, 'portfolio_value': final_value, 'action': dominant_action, 'strategy': 'TopKFusion', 'fusion_allocation': allocation}
        self.episode_rewards.append(reward)
        self.episode_actions.append(dominant_action)
        self.episode_returns.append(total_return)
        self.episode_sharpes.append(sharpe_ratio)
        self.episode_trades.append(num_trades)
        self.episode_infos.append(info)
        self.episode_portfolio_values.append(final_value)
        return (self._get_observation(), reward, done, info)

    def _compute_window_price_change(self, exec_start: int, exec_end: int) -> float:
        try:
            n = len(self.df)
            s = min(exec_start, n - 1)
            e = min(exec_end - 1, n - 1)
            if s >= e:
                return 0.0
            p_start = float(self.df['close'].iat[s])
            p_end = float(self.df['close'].iat[e])
            if p_start > 0 and np.isfinite(p_start) and np.isfinite(p_end):
                return (p_end - p_start) / p_start
        except Exception:
            pass
        return 0.0

    def _compute_max_drawdown(self, portfolio_values) -> float:
        pv = np.array(portfolio_values, dtype=np.float64)
        if len(pv) < 2:
            return 0.0
        running_max = np.maximum.accumulate(pv)
        drawdowns = np.where(running_max > 0, (running_max - pv) / running_max, 0.0)
        return float(np.max(drawdowns))

    def _compute_reward(self, total_return: float, sharpe_ratio: float) -> float:
        return_weight = self.reward_config.get('return_weight', 0.7)
        sharpe_weight = self.reward_config.get('sharpe_weight', 0.3)
        scale = self.reward_config.get('reward_scale_factor', 1.0)
        reward = (return_weight * total_return + sharpe_weight * sharpe_ratio) * scale
        reward = np.clip(reward, -5.0, 5.0)
        return float(reward)

    def _compute_true_sharpe(self, equity_curve) -> float:
        eq = np.array(equity_curve, dtype=np.float64)
        if len(eq) < 3:
            return 0.0
        prev = eq[:-1]
        daily_ret = np.where(prev > 0, (eq[1:] - prev) / prev, 0.0)
        rf_daily = 0.0
        excess = daily_ret - rf_daily
        if len(excess) < 2:
            return 0.0
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        sharpe = float(np.mean(excess)) / std * np.sqrt(252)
        return float(np.clip(sharpe, -10.0, 10.0))

    def get_episode_summary(self) -> Dict:
        n = len(self.episode_rewards)
        if n == 0:
            return {'total_reward': 0.0, 'avg_reward': 0.0, 'avg_return': 0.0, 'avg_sharpe': 0.0, 'avg_trades': 0, 'max_drawdown': 0.0, 'steps': 0, 'actions': []}
        if self.sequential_eval:
            compounded = 1.0
            for r in self.episode_returns:
                compounded *= 1.0 + r
            avg_return = float(compounded - 1.0)
            if self.episode_daily_equity and len(self.episode_daily_equity) >= 2:
                max_drawdown = self._compute_max_drawdown(self.episode_daily_equity)
            elif self.episode_portfolio_values:
                first_return = self.episode_returns[0]
                denom = 1.0 + first_return if abs(1.0 + first_return) > 1e-10 else 1.0
                initial_val = self.episode_portfolio_values[0] / denom
                pv_series = [initial_val] + list(self.episode_portfolio_values)
                max_drawdown = self._compute_max_drawdown(pv_series)
            else:
                max_drawdown = 0.0
            if self.episode_daily_equity and len(self.episode_daily_equity) >= 3:
                avg_sharpe = self._compute_true_sharpe(self.episode_daily_equity)
            elif self.episode_portfolio_values and len(self.episode_portfolio_values) >= 2:
                first_return = self.episode_returns[0]
                denom = 1.0 + first_return if abs(1.0 + first_return) > 1e-10 else 1.0
                initial_val = self.episode_portfolio_values[0] / denom
                pv_series = [initial_val] + list(self.episode_portfolio_values)
                avg_sharpe = self._compute_true_sharpe(pv_series)
            else:
                avg_sharpe = float(np.mean(self.episode_sharpes))
        else:
            avg_return = float(np.mean(self.episode_returns))
            if self.episode_daily_equity and len(self.episode_daily_equity) >= 2:
                max_drawdown = self._compute_max_drawdown(self.episode_daily_equity)
            else:
                cum_vals = [1.0]
                for r in self.episode_returns:
                    cum_vals.append(cum_vals[-1] * (1.0 + r))
                max_drawdown = self._compute_max_drawdown(cum_vals)
            avg_sharpe = float(np.mean(self.episode_sharpes))
        return {'total_reward': float(np.sum(self.episode_rewards)), 'avg_reward': float(np.mean(self.episode_rewards)), 'avg_return': avg_return, 'avg_sharpe': avg_sharpe, 'avg_trades': float(np.mean(self.episode_trades)), 'max_drawdown': max_drawdown, 'steps': n, 'actions': list(self.episode_actions)}
