import os
import random
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import numpy as _np_json

class _NumpyJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, _np_json.integer):
            return int(obj)
        if isinstance(obj, _np_json.floating):
            return float(obj)
        if isinstance(obj, _np_json.ndarray):
            return obj.tolist()
        return super().default(obj)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str, name: str='rl_training') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'train_{timestamp}.log'), encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class RewardNormalizer:

    def __init__(self, clip_range: Optional[tuple]=(-10.0, 10.0), momentum: float=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.momentum = momentum
        self.clip_range = clip_range

    def update(self, rewards: np.ndarray):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
        self.count += len(rewards)

    def normalize(self, reward: float) -> float:
        if self.var > 1e-08:
            normalized = (reward - self.mean) / (np.sqrt(self.var) + 1e-08)
        else:
            normalized = reward - self.mean
        if self.clip_range is not None:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        return normalized

    def state_dict(self) -> Dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state: Dict):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']

class CheckpointManager:

    def __init__(self, save_dir: str, max_keep: int=5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(save_dir, exist_ok=True)
        self.best_metric = -float('inf')
        self.checkpoints = []

    def save_checkpoint(self, state: Dict[str, Any], is_best: bool=False, filename: str='checkpoint.pt'):
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(state, best_path)
        if not is_best and filename not in ['last_model.pt', 'best_model.pt']:
            self.checkpoints.append(filepath)
            if len(self.checkpoints) > self.max_keep:
                old_ckpt = self.checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Checkpoint not found: {filepath}')
        return torch.load(filepath, map_location='cpu', weights_only=False)

    def update_best(self, metric: float) -> bool:
        if metric > self.best_metric:
            self.best_metric = metric
            return True
        return False

class MetricsTracker:

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_history = {'train': [], 'val': [], 'val_greedy': [], 'val_stochastic': [], 'test': [], 'test_greedy': [], 'test_stochastic': []}
        os.makedirs(log_dir, exist_ok=True)

    def log_metrics(self, metrics: Dict[str, Any], phase: str, step: int):
        if phase not in self.metrics_history:
            self.metrics_history[phase] = []
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history[phase].append(metrics)

    def save_metrics(self):
        filepath = os.path.join(self.log_dir, 'metrics_history.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False, cls=_NumpyJsonEncoder)

    def get_best_val_metric(self, metric_name: str='avg_return') -> float:
        if not self.metrics_history['val']:
            return -float('inf')
        return max((m[metric_name] for m in self.metrics_history['val'] if metric_name in m))

def linear_schedule(initial_value: float, final_value: float, current_step: int, total_steps: int) -> float:
    fraction = min(1.0, current_step / total_steps)
    return initial_value + fraction * (final_value - initial_value)

def cosine_schedule(initial_value: float, final_value: float, current_step: int, total_steps: int) -> float:
    fraction = min(1.0, current_step / total_steps)
    return final_value + 0.5 * (initial_value - final_value) * (1 + np.cos(np.pi * fraction))

def compute_strategy_distribution(actions: list, num_strategies: int=11) -> Dict[str, float]:
    counts = np.bincount(actions, minlength=num_strategies)
    total = len(actions)
    distribution = {f'strategy_{i}': counts[i] / total for i in range(num_strategies)}
    return distribution

def print_training_header(config: Dict[str, Any]):
    print('=' * 80)
    print('Two-stage PPO training system')
    print('=' * 80)
    print(f':')
    for key, value in config.items():
        print(f'  {key}: {value}')
    print('=' * 80)
import pandas as _pd

def fix_forward_returns(df: _pd.DataFrame) -> _pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['ts_code', 'trade_date'])
    for col, periods in [('ret_1d', 1), ('ret_5d', 5), ('ret_20d', 20)]:
        if col in df.columns:
            df[col] = df.groupby('ts_code', group_keys=False)['close'].transform(lambda x: x.pct_change(periods=periods))
    return df
