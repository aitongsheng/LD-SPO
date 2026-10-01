import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict
from ldspo_config import NUMERICAL_FEATURE_COLS, STRATEGY_ACTIONS, TEACHER_COLUMN_ALIASES
from multimodal_observation import build_news_observation
from market_data import load_table
TEACHER_COLS = [TEACHER_COLUMN_ALIASES[i][0] for i in range(len(STRATEGY_ACTIONS))]
NUM_ACTIONS = len(TEACHER_COLS)
_PRICE_COLS = frozenset({'open', 'high', 'low', 'close', 'ma_5', 'ma_10', 'ma_20', 'ma_60', 'boll_up', 'boll_down', 'change', 'atr_14'})
_VOLUME_COLS = frozenset({'vol', 'amount', 'vol_20d'})

class BCStockDataset(Dataset):

    def __init__(self, df: pd.DataFrame, lookback: int=60, action_indices: Optional[List[int]]=None):
        self.lookback = lookback
        self.df = df.reset_index(drop=True)
        self.action_indices = action_indices
        self.feature_cols = [c for c in NUMERICAL_FEATURE_COLS if c in self.df.columns]
        self._build_feature_matrix()
        self._build_teacher_matrix()
        self._build_value_matrix()
        self.valid_indices = list(range(lookback - 1, len(self.df)))

    def _build_feature_matrix(self):
        raw = self.df[self.feature_cols].copy().astype(np.float32)
        if 'close' in self.df.columns:
            close_vals = self.df['close'].values.astype(np.float32)
            close_vals = np.where(close_vals > 0, close_vals, 1.0)
            for col in self.feature_cols:
                if col in _PRICE_COLS and col != 'close':
                    raw[col] = raw[col].values / close_vals
            raw['close'] = 1.0
        for col in self.feature_cols:
            if col in _VOLUME_COLS:
                rolling_mean = self.df[col].rolling(window=20, min_periods=1).mean().values.astype(np.float32)
                rolling_mean = np.where(rolling_mean > 0, rolling_mean, 1.0)
                raw[col] = raw[col].values / rolling_mean
        self.feature_matrix = raw.fillna(0.0).values
        self.num_features = self.feature_matrix.shape[1]

    def _build_value_matrix(self):
        if 'teacher_value_target' not in self.df.columns:
            self.has_value_target = False
            self.value_matrix = np.zeros(len(self.df), dtype=np.float32)
            return
        vt = self.df['teacher_value_target'].fillna(0.0).values.astype(np.float32)
        valid_mask = ~np.isnan(vt)
        if valid_mask.sum() > 1:
            mu = vt[valid_mask].mean()
            std = vt[valid_mask].std()
            if std > 1e-08:
                vt = np.where(valid_mask, (vt - mu) / std, 0.0)
        self.value_matrix = vt.astype(np.float32)
        self.has_value_target = True

    def _build_teacher_matrix(self):
        if self.action_indices is not None:
            selected_indices = self.action_indices
            effective_n = len(self.action_indices)
        else:
            selected_indices = list(range(NUM_ACTIONS))
            effective_n = NUM_ACTIONS
        resolved_cols = []
        for action_idx in selected_indices:
            resolved_cols.append(next((c for c in TEACHER_COLUMN_ALIASES[action_idx] if c in self.df.columns), None))
        if not any(resolved_cols):
            self.teacher_matrix = np.full((len(self.df), effective_n), 1.0 / effective_n, dtype=np.float32)
            return
        teacher_data = np.zeros((len(self.df), effective_n), dtype=np.float32)
        for i, col in enumerate(resolved_cols):
            if col is not None:
                teacher_data[:, i] = self.df[col].fillna(0.0).values.astype(np.float32)
        row_sums = teacher_data.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-08, row_sums, 1.0)
        self.teacher_matrix = teacher_data / row_sums

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple:
        t = self.valid_indices[idx]
        start = max(0, t - self.lookback + 1)
        obs = self.feature_matrix[start:t + 1]
        if obs.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - obs.shape[0], self.num_features), dtype=np.float32)
            obs = np.concatenate([pad, obs], axis=0)
        teacher = self.teacher_matrix[t]
        news_token_ids, news_mask = build_news_observation(self.df, t)
        multimodal_obs = {'numerical': torch.tensor(obs, dtype=torch.float32), 'news_token_ids': torch.tensor(news_token_ids, dtype=torch.long), 'news_mask': torch.tensor(news_mask, dtype=torch.bool)}
        if self.has_value_target:
            value_target = self.value_matrix[t]
            return (multimodal_obs, torch.tensor(teacher, dtype=torch.float32), torch.tensor(value_target, dtype=torch.float32))
        return (multimodal_obs, torch.tensor(teacher, dtype=torch.float32))

class BCDataset(Dataset):

    def __init__(self, df: pd.DataFrame, lookback: int=60, date_start: Optional[str]=None, date_end: Optional[str]=None, min_stock_days: int=0, verbose: bool=True, action_indices: Optional[List[int]]=None):
        self.lookback = lookback
        self.action_indices = action_indices
        if 'trade_date' in df.columns:
            df = df.copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            if date_start:
                df = df[df['trade_date'] >= date_start]
            if date_end:
                df = df[df['trade_date'] <= date_end]
        self.numerical_list: List[np.ndarray] = []
        self.news_token_ids_list: List[np.ndarray] = []
        self.news_mask_list: List[np.ndarray] = []
        self.teacher_list: List[np.ndarray] = []
        self.value_list: List[float] = []
        self.has_value_target: bool = False
        if 'ts_code' in df.columns:
            stock_groups = df.groupby('ts_code')
        elif 'symbol' in df.columns:
            stock_groups = df.groupby('symbol')
        else:
            stock_groups = [('ALL', df)]
        n_stocks_used = 0
        n_skipped = 0
        for stock_code, stock_df in stock_groups:
            stock_df = stock_df.sort_values('trade_date').reset_index(drop=True)
            if min_stock_days > 0 and len(stock_df) < min_stock_days:
                n_skipped += 1
                continue
            if len(stock_df) < lookback + 1:
                n_skipped += 1
                continue
            ds = BCStockDataset(stock_df, lookback=lookback, action_indices=action_indices)
            if ds.has_value_target:
                self.has_value_target = True
            for i in range(len(ds)):
                item = ds[i]
                self.numerical_list.append(item[0]['numerical'].numpy())
                self.news_token_ids_list.append(item[0]['news_token_ids'].numpy().astype(np.int32))
                self.news_mask_list.append(item[0]['news_mask'].numpy())
                self.teacher_list.append(item[1].numpy())
                if ds.has_value_target:
                    self.value_list.append(float(item[2]))
                else:
                    self.value_list.append(0.0)
            n_stocks_used += 1
        if len(self.numerical_list) == 0:
            raise ValueError(f'[BCDataset] !Date range ({date_start} ~ {date_end})  lookback ({lookback}) . {n_skipped} , {n_stocks_used} ().')
        self.obs_array = np.stack(self.numerical_list, axis=0)
        self.news_token_ids_array = np.stack(self.news_token_ids_list, axis=0)
        self.news_mask_array = np.stack(self.news_mask_list, axis=0)
        self.teacher_array = np.stack(self.teacher_list, axis=0)
        self.value_array = np.array(self.value_list, dtype=np.float32)
        effective_indices = action_indices if action_indices is not None else list(range(NUM_ACTIONS))
        if verbose:
            print(f'[BCDataset] Date range: {date_start} ~ {date_end}')
            print(f'[BCDataset] : {n_stocks_used}, : {n_skipped}')
            print(f'[BCDataset] Total samples: {len(self.obs_array)}')
            print(f'[BCDataset] Feature shape: {self.obs_array.shape}')
            print(f'[BCDataset] News tensors: ids={self.news_token_ids_array.shape}, mask={self.news_mask_array.shape}')
            print(f"[BCDataset] Value Target: {('Loaded' if self.has_value_target else '( value loss)')}")
            print(f"[BCDataset] Action space: {len(effective_indices)}  {[STRATEGY_ACTIONS[i]['name'] for i in effective_indices]}")
            mean_labels = self.teacher_array.mean(axis=0)
            print(f'[BCDataset] :')
            for local_i, orig_i in enumerate(effective_indices):
                name = STRATEGY_ACTIONS[orig_i]['name']
                print(f'            [{local_i:2d}→{orig_i:2d}] {name:12s}: {mean_labels[local_i]:.4f}')

    def __len__(self) -> int:
        return len(self.obs_array)

    def __getitem__(self, idx: int) -> Tuple:
        return ({'numerical': torch.tensor(self.obs_array[idx], dtype=torch.float32), 'news_token_ids': torch.tensor(self.news_token_ids_array[idx], dtype=torch.long), 'news_mask': torch.tensor(self.news_mask_array[idx], dtype=torch.bool)}, torch.tensor(self.teacher_array[idx], dtype=torch.float32), torch.tensor(self.value_array[idx], dtype=torch.float32))

def build_dataloader(df: pd.DataFrame, lookback: int=60, date_start: Optional[str]=None, date_end: Optional[str]=None, batch_size: int=256, shuffle: bool=True, num_workers: int=4, verbose: bool=True) -> Tuple[DataLoader, BCDataset]:
    dataset = BCDataset(df=df, lookback=lookback, date_start=date_start, date_end=date_end, verbose=verbose)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)
    return (loader, dataset)
if __name__ == '__main__':
    import pandas as pd
    bundle_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = load_table(os.path.join(bundle_root, 'data', 'us_stocks_ohlcv_with_features_and_teachers.parquet'))
    loader, dataset = build_dataloader(df=df, lookback=60, date_start='2018-01-01', date_end='2020-12-31', batch_size=256, shuffle=True, num_workers=0)
    print(f'\nDataset size: {len(dataset)}')
    print(f'Batches: {len(loader)}')
    obs, teacher = next(iter(loader))
    print(f'obs shape: {obs.shape}')
    print(f'teacher shape: {teacher.shape}')
    print(f'teacher 3:\n{teacher[:3]}')
    print(f'teacher : {teacher[:3].sum(dim=-1)}')
