import numpy as np
import torch
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import time
import traceback

def _pool_initializer():
    try:
        import pandas as _pd
        from trading_pomdp import RLTradingEnvWrapper as _Env
        from ldspo_model import LDSPOMultimodalStudent
        import torch as _torch
        _torch.set_num_threads(2)
    except Exception:
        pass

def _worker_rollout(args_tuple):
    stock_code, stock_df_dict, args_dict, reward_config, model_state_numpy, model_config, rollout_steps = args_tuple
    try:
        import pandas as pd
        import torch
        import numpy as np
        from trading_pomdp import RLTradingEnvWrapper
        from ldspo_model import LDSPOMultimodalStudent
        stock_df = pd.DataFrame(stock_df_dict)
        if 'trade_date' in stock_df.columns:
            stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        env = RLTradingEnvWrapper(df=stock_df, window_size=args_dict['window_size'], lookback=args_dict['lookback'], start_idx=0, end_idx=len(stock_df), reward_config=reward_config, allow_short=args_dict.get('allow_short', False))
        model_type = model_config.get('model_type', 'multimodal')
        num_features = model_config['num_features']
        num_actions = model_config.get('num_actions', 11)
        seq_len = args_dict['lookback']
        if model_type != 'multimodal':
            raise ValueError('Only the paper-aligned multimodal model is included')
        model = LDSPOMultimodalStudent(numerical_dim=num_features, seq_len=seq_len, d_model=model_config.get('d_model', 128), n_heads=model_config.get('n_heads', 4), n_layers=model_config.get('n_layers', 3), num_actions=num_actions)
        model_state_dict = {k: torch.from_numpy(v) for k, v in model_state_numpy.items()}
        model.load_state_dict(model_state_dict)
        model.eval()
        temperature = model_config.get('temperature', 1.0)
        action_mapping = args_dict.get('action_mapping', None)
        obs = env.reset(max_steps=rollout_steps)
        trajectory = []
        for step in range(rollout_steps):
            numerical = torch.tensor(obs['numerical'], dtype=torch.float32).unsqueeze(0)
            model_input = {'numerical': numerical, 'news_token_ids': torch.tensor(obs['news_token_ids'], dtype=torch.long).unsqueeze(0), 'news_mask': torch.tensor(obs['news_mask'], dtype=torch.bool).unsqueeze(0)}
            with torch.no_grad():
                action_t, log_prob_t, value_t, entropy_t = model.get_action(model_input, deterministic=False, temperature=temperature)
            action = action_t.item()
            log_prob = log_prob_t.item()
            value = value_t.item()
            real_action = action_mapping[action] if action_mapping is not None else action
            obs_next, reward, done, info = env.step(real_action)
            safe_info = {k: v for k, v in info.items() if isinstance(v, (int, float, bool, str, type(None)))}
            trajectory.append({'obs': obs, 'action': action, 'reward': reward, 'value': value, 'log_prob': log_prob, 'done': done, 'step': step, 'info': safe_info})
            obs = obs_next
            if done:
                break
        summary = env.get_episode_summary()
        summary['stock_code'] = stock_code
        return (summary, trajectory)
    except Exception as e:
        print(f'Worker error for {stock_code}: {e}')
        traceback.print_exc()
        return ({'total_reward': 0.0, 'avg_reward': 0.0, 'avg_return': 0.0, 'avg_sharpe': 0.0, 'avg_trades': 0, 'steps': 0, 'stock_code': stock_code, 'error': str(e), 'actions': []}, [])

class ParallelSampler:

    def __init__(self, num_workers: int=4, args_dict: Optional[Dict]=None, reward_config: Optional[Dict]=None, logger=None):
        self.num_workers = num_workers
        self.args_dict = args_dict or {}
        self.reward_config = reward_config or {}
        self.logger = logger
        self.pool = None

    def _ensure_pool(self):
        if self.pool is None:
            self.pool = mp.Pool(processes=self.num_workers, initializer=_pool_initializer)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close()

def parallel_rollout_episode(sampler: ParallelSampler, agent, stock_df_list: List[Tuple[str, 'pd.DataFrame']], rollout_steps: int, logger=None, verbose: bool=False, pre_cpu_state: Optional[Dict]=None, pre_model_config: Optional[Dict]=None, pre_df_dicts: Optional[Dict]=None, worker_timeout: int=600) -> Tuple[List[Dict], List[List]]:
    sampler._ensure_pool()
    if pre_cpu_state is not None and pre_model_config is not None:
        _sample = next(iter(pre_cpu_state.values()), None)
        if _sample is not None and isinstance(_sample, torch.Tensor):
            cpu_state_numpy = {k: v.detach().cpu().numpy() for k, v in pre_cpu_state.items()}
        else:
            cpu_state_numpy = pre_cpu_state
        model_config = pre_model_config
    else:
        with torch.no_grad():
            cpu_state_numpy = {k: v.detach().cpu().numpy() for k, v in agent.model.state_dict().items()}
        model_config = {'model_type': getattr(agent, 'model_type', 'multimodal'), 'num_features': agent.obs_shape['numerical'][1], 'num_actions': agent.num_actions, 'd_model': getattr(agent, 'd_model', 128), 'n_heads': getattr(agent, 'n_heads', 4), 'n_layers': getattr(agent, 'n_layers', 3), 'hidden_dim': 256, 'temperature': getattr(agent, 'temperature', 1.0)}
    worker_args = []
    for stock_code, stock_df in stock_df_list:
        if pre_df_dicts is not None and stock_code in pre_df_dicts:
            df_dict = pre_df_dicts[stock_code]
        else:
            df_dict = {}
            for col in stock_df.columns:
                if stock_df[col].dtype == 'datetime64[ns]':
                    df_dict[col] = stock_df[col].dt.strftime('%Y-%m-%d').tolist()
                else:
                    df_dict[col] = stock_df[col].tolist()
        worker_args.append((stock_code, df_dict, sampler.args_dict, sampler.reward_config, cpu_state_numpy, model_config, rollout_steps))
    t_start = time.time()
    try:
        async_result = sampler.pool.map_async(_worker_rollout, worker_args)
        results = async_result.get(timeout=worker_timeout)
    except mp.TimeoutError:
        err_msg = f'Parallel rollout timeout(>{worker_timeout}s), batch'
        if logger:
            logger.error(f'❌ {err_msg}')
        else:
            print(f'❌ {err_msg}')
        results = [({'total_reward': 0.0, 'avg_reward': 0.0, 'avg_return': 0.0, 'avg_sharpe': 0.0, 'avg_trades': 0, 'steps': 0, 'stock_code': sc, 'error': 'timeout', 'actions': []}, []) for sc, _ in stock_df_list]
    except Exception as e:
        err_msg = f'Parallel rollout error: {e}'
        if logger:
            logger.error(f'❌ {err_msg}')
        else:
            print(f'❌ {err_msg}')
        traceback.print_exc()
        results = [({'total_reward': 0.0, 'avg_reward': 0.0, 'avg_return': 0.0, 'avg_sharpe': 0.0, 'avg_trades': 0, 'steps': 0, 'stock_code': sc, 'error': str(e), 'actions': []}, []) for sc, _ in stock_df_list]
    t_elapsed = time.time() - t_start
    if verbose and logger:
        logger.info(f'    Parallel samplingcompleted: {len(results)} ,  {t_elapsed:.1f}s')
    all_summaries = []
    all_trajectories = []
    for summary, trajectory in results:
        all_summaries.append(summary)
        all_trajectories.append(trajectory)
        for step_data in trajectory:
            agent.buffer.add(obs=step_data['obs'], action=step_data['action'], reward=step_data['reward'], value=step_data['value'], log_prob=step_data['log_prob'], done=step_data['done'])
    return (all_summaries, all_trajectories)
