from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data' / 'sample' / 'ldspo_sample.csv'
STRATEGIES = ['Weave', 'Inertia', 'Breach', 'Flux', 'Envelope', 'Reversion', 'Correction', 'Oscillation', 'Long', 'Short', 'Cash']

def build_stock(symbol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=180)
    returns = rng.normal(0.0004, 0.014, len(dates))
    close = 100.0 * np.exp(np.cumsum(returns))
    open_price = close * (1.0 + rng.normal(0.0, 0.003, len(dates)))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.012, len(dates)))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.012, len(dates)))
    volume = rng.lognormal(14.0, 0.3, len(dates))
    frame = pd.DataFrame({'trade_date': dates, 'ts_code': symbol, 'open': open_price, 'high': high, 'low': low, 'close': close, 'vol': volume})
    frame['amount'] = frame['close'] * frame['vol']
    frame['change'] = frame['close'].diff().fillna(0.0)
    frame['pct_chg'] = frame['close'].pct_change().fillna(0.0) * 100.0
    frame['ret_1d'] = frame['close'].pct_change().fillna(0.0)
    frame['ret_5d'] = frame['close'].pct_change(5).fillna(0.0)
    frame['ret_20d'] = frame['close'].pct_change(20).fillna(0.0)
    for window in (5, 10, 20, 60):
        frame[f'ma_{window}'] = frame['close'].rolling(window, min_periods=1).mean()
    frame['price_ma_20'] = frame['close'] / frame['ma_20']
    frame['ma_20_vs_60'] = frame['ma_20'] / frame['ma_60']
    rolling_std = frame['close'].rolling(20, min_periods=2).std().fillna(1.0)
    frame['bb_zscore'] = ((frame['close'] - frame['ma_20']) / rolling_std).fillna(0.0)
    frame['vol_20d'] = frame['vol'].rolling(20, min_periods=1).mean()
    volume_std = frame['vol'].rolling(20, min_periods=2).std().replace(0.0, np.nan)
    frame['volume_zscore'] = ((frame['vol'] - frame['vol_20d']) / volume_std).fillna(0.0)
    true_range = np.maximum.reduce([(frame['high'] - frame['low']).to_numpy(), (frame['high'] - frame['close'].shift(1)).abs().fillna(0.0).to_numpy(), (frame['low'] - frame['close'].shift(1)).abs().fillna(0.0).to_numpy()])
    frame['atr_14'] = pd.Series(true_range).rolling(14, min_periods=1).mean().to_numpy()
    frame['stock_news_text'] = [f'{symbol} synthetic company update for {date.date()}' if index % 3 == 0 else '' for index, date in enumerate(dates)]
    frame['sector_news_text'] = ['Synthetic sector demand update' if index % 5 == 0 else '' for index in range(len(dates))]
    frame['market_news_text'] = ['Synthetic market liquidity update' if index % 7 == 0 else '' for index in range(len(dates))]
    frame['macro_news_text'] = ['Synthetic macro policy update' if index % 11 == 0 else '' for index in range(len(dates))]
    teacher = rng.dirichlet(np.ones(len(STRATEGIES)) * 1.5, size=len(dates))
    for index, name in enumerate(STRATEGIES):
        frame[f'teacher_action_{name}'] = teacher[:, index]
    frame['teacher_value_target'] = np.tanh(frame['ret_20d'] * 10.0)
    return frame

def main() -> None:
    sample = pd.concat([build_stock('SAMPLE_A', 42), build_stock('SAMPLE_B', 314)], ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT, index=False)
    print(f'Wrote {len(sample)} rows to {OUTPUT}')
if __name__ == '__main__':
    main()
