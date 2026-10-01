import os
_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    if not os.environ.get(_var, '1').strip() or os.environ.get(_var, '1') == '0':
        os.environ[_var] = '1'
import argparse
import time
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from knowledge_distillation import pretrain as run_pretrain
from policy_optimization import PPOAgentV2
from stage_aware_kl import KL_SCHEDULES, get_kl_coef, validate_kl_schedule_args
from trading_pomdp import RLTradingEnvWrapper
from on_policy_collection import ParallelSampler, parallel_rollout_episode
from ldspo_config import STRATEGY_ACTIONS, NUMERICAL_FEATURE_COLS
from experiment_utils import set_seed, setup_logger, CheckpointManager, MetricsTracker, compute_strategy_distribution, print_training_header
from market_data import load_table

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_only', action='store_true')
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=os.path.join(_BUNDLE_ROOT, 'data', 'sample', 'ldspo_sample.csv'))
    parser.add_argument('--train_start', type=str, default='2023-01-02')
    parser.add_argument('--train_end', type=str, default='2023-05-31')
    parser.add_argument('--val_start', type=str, default='2023-06-01')
    parser.add_argument('--val_end', type=str, default='2023-07-14')
    parser.add_argument('--test_start', type=str, default='2023-07-17')
    parser.add_argument('--test_end', type=str, default='2023-09-08')
    parser.add_argument('--min_data_days', type=int, default=None)
    parser.add_argument('--lookback', type=int, default=30)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--model_type', type=str, default='multimodal', choices=['multimodal'])
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--s1_n_epochs', type=int, default=30)
    parser.add_argument('--s1_batch_size', type=int, default=512)
    parser.add_argument('--s1_lr', type=float, default=0.0003)
    parser.add_argument('--s1_weight_decay', type=float, default=0.0001)
    parser.add_argument('--s1_entropy_coef', type=float, default=0.01)
    parser.add_argument('--s1_warmup_ratio', type=float, default=0.05)
    parser.add_argument('--s1_max_grad_norm', type=float, default=1.0)
    parser.add_argument('--s1_num_workers', type=int, default=4)
    parser.add_argument('--s1_unfreeze_value_head_epoch', type=int, default=20)
    parser.add_argument('--s1_value_loss_coef', type=float, default=0.5)
    parser.add_argument('--s1_value_head_lr_ratio', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.1)
    parser.add_argument('--entropy_coef_final', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--value_clip_range', type=float, default=0.2)
    parser.add_argument('--normalize_returns', action='store_true', default=True)
    parser.add_argument('--target_kl', type=float, default=0.15)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--temperature_final', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--kl_coef', type=float, default=0.03)
    parser.add_argument('--kl_coef_final', type=float, default=0.0)
    parser.add_argument('--kl_warmup_episodes', type=int, default=50)
    parser.add_argument('--kl_schedule', type=str, default='linear', choices=KL_SCHEDULES)
    parser.add_argument('--n_episodes', type=int, default=300)
    parser.add_argument('--rollout_steps', type=int, default=128)
    parser.add_argument('--stocks_per_episode', type=int, default=16)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--eval_mode', type=str, default='both', choices=['greedy', 'topk', 'both'])
    parser.add_argument('--entropy_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--reward_return_weight', type=float, default=0.7)
    parser.add_argument('--reward_sharpe_weight', type=float, default=0.3)
    parser.add_argument('--reward_scale_factor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default=os.path.join(_BUNDLE_ROOT, 'outputs', 'manual'))
    parser.add_argument('--allow_short', action='store_true', default=False)
    return parser.parse_args()

def _patch_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'ts_code' not in df.columns and 'symbol' in df.columns:
        df['ts_code'] = df['symbol']
    if 'vol' not in df.columns and 'volume' in df.columns:
        df['vol'] = df['volume']
    if 'amount' not in df.columns:
        if 'vwap' in df.columns and 'volume' in df.columns:
            df['amount'] = df['vwap'] * df['volume']
        elif 'vol' in df.columns and 'close' in df.columns:
            df['amount'] = df['vol'] * df['close']
    if 'pct_chg' not in df.columns and 'ret_1d' in df.columns:
        df['pct_chg'] = df['ret_1d']
    if 'change' not in df.columns and 'close' in df.columns and ('ret_1d' in df.columns):
        df['change'] = df['close'] * df['ret_1d']
    return df

def load_and_split_data(args, logger=None):
    df = load_table(args.data_path)
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    else:
        raise ValueError("Missing 'trade_date' ")
    df = _patch_df_columns(df)
    if 'ts_code' not in df.columns:
        raise ValueError("Missing: 'ts_code'  'symbol' ")
    train_df = df[(df['trade_date'] >= args.train_start) & (df['trade_date'] <= args.train_end)].copy()
    val_df = df[(df['trade_date'] >= args.val_start) & (df['trade_date'] <= args.val_end)].copy()
    test_df = df[(df['trade_date'] >= args.test_start) & (df['trade_date'] <= args.test_end)].copy()
    if args.min_data_days is None:
        min_days = args.lookback + args.rollout_steps + args.window_size
    else:
        min_days = args.min_data_days

    def _split_by_stock(subset_df, split_name):
        stocks = {}
        skipped = 0
        for code in subset_df['ts_code'].unique():
            sdf = subset_df[subset_df['ts_code'] == code].sort_values('trade_date').reset_index(drop=True)
            if len(sdf) >= min_days:
                stocks[code] = sdf
            else:
                skipped += 1
        if logger:
            logger.info(f'{split_name}: {len(stocks)} , {len(subset_df)}  ( {skipped} )')
        return stocks
    train_stocks = _split_by_stock(train_df, 'training')
    val_stocks = _split_by_stock(val_df, 'validation')
    test_stocks = _split_by_stock(test_df, 'test')
    return (train_stocks, val_stocks, test_stocks)

def create_env(stock_df, args, reward_config, sequential_eval=False):
    return RLTradingEnvWrapper(df=stock_df, window_size=args.window_size, lookback=args.lookback, start_idx=0, end_idx=len(stock_df), reward_config=reward_config, sequential_eval=sequential_eval, allow_short=getattr(args, 'allow_short', False))

def create_reward_config(args):
    return {'return_weight': getattr(args, 'reward_return_weight', 0.7), 'sharpe_weight': getattr(args, 'reward_sharpe_weight', 0.3), 'reward_scale_factor': getattr(args, 'reward_scale_factor', 1.0)}

def rollout_episode(env, agent, max_steps=None, deterministic=False, stock_code=None, execution_mode='sample', top_k=3):
    obs = env.reset(max_steps=max_steps)
    done = False
    step_count = 0
    trajectory = []
    start_idx = env.current_step
    start_date = None
    end_date = None
    if hasattr(env, 'df') and 'trade_date' in env.df.columns and (start_idx < len(env.df)):
        start_date = env.df.iloc[start_idx]['trade_date']
    while not done:
        if execution_mode == 'topk':
            probs, value = agent.policy_distribution(obs)
            k = min(top_k, len(probs))
            top_actions = np.argsort(probs)[-k:][::-1]
            top_weights = probs[top_actions]
            top_weights = top_weights / max(top_weights.sum(), 1e-12)
            action = int(top_actions[0])
            log_prob = float(np.log(max(probs[action], 1e-12)))
            obs_next, reward, done, info = env.step_fusion(top_actions, top_weights)
        else:
            action, log_prob, value = agent.select_action(obs, deterministic=deterministic)
            obs_next, reward, done, info = env.step(action)
        if not deterministic and execution_mode != 'topk':
            agent.buffer.add(obs, action, reward, value, log_prob, done)
        trajectory.append({'step': step_count, 'action': action, 'reward': reward, 'value': value, 'info': info})
        obs = obs_next
        step_count += 1
        if max_steps and step_count >= max_steps:
            break
    end_idx = env.current_step
    if hasattr(env, 'df') and 'trade_date' in env.df.columns and (end_idx < len(env.df)):
        end_date = env.df.iloc[end_idx]['trade_date']
    summary = env.get_episode_summary()
    summary.update({'steps': step_count, 'stock_code': stock_code, 'start_idx': start_idx, 'end_idx': end_idx, 'start_date': start_date, 'end_date': end_date})
    return (summary, trajectory)

def evaluate(env, agent, n_episodes=5, eval_mode='both'):

    def _eval_once(mode):
        returns, sharpes, trades, rewards, actions, drawdowns = ([], [], [], [], [], [])
        for _ in range(n_episodes):
            summary, _ = rollout_episode(env, agent, deterministic=mode == 'greedy', execution_mode='topk' if mode == 'topk' else 'sample')
            returns.append(summary.get('avg_return', 0.0))
            sharpes.append(summary.get('avg_sharpe', 0.0))
            trades.append(summary.get('avg_trades', 0))
            rewards.append(summary.get('avg_reward', 0.0))
            actions.extend(summary.get('actions', []))
            drawdowns.append(summary.get('max_drawdown', 0.0))
        dist = compute_strategy_distribution(actions, num_strategies=11)
        dist_vals = np.array([dist.get(f'strategy_{i}', 0.0) for i in range(11)]) + 1e-10
        dist_vals /= dist_vals.sum()
        entropy = -np.sum(dist_vals * np.log(dist_vals))
        return {'avg_return': np.mean(returns), 'std_return': np.std(returns), 'avg_sharpe': np.mean(sharpes), 'avg_trades': np.mean(trades), 'avg_reward': np.mean(rewards), 'avg_max_drawdown': np.mean(drawdowns), 'strategy_distribution': dist, 'action_entropy': entropy}
    if eval_mode == 'greedy':
        return _eval_once('greedy')
    elif eval_mode == 'topk':
        return _eval_once('topk')
    else:
        return {'greedy': _eval_once('greedy'), 'topk': _eval_once('topk')}

def print_action_dist(actions, episode, logger):
    if not actions:
        return
    counts = Counter(actions)
    total = len(actions)
    lines = [f"\n{'=' * 70}", f'📊 Episode {episode} - Action distribution ( {total} )', f"{'=' * 70}"]
    for i in range(11):
        cnt = counts.get(i, 0)
        if cnt > 0:
            pct = cnt / total * 100
            bar = '█' * int(pct / 3)
            name = STRATEGY_ACTIONS[i]['name']
            lines.append(f'  [{i:2d}] {name:12s}: {cnt:5d} ({pct:5.1f}%) {bar}')
    probs = np.array([counts.get(i, 0) / total for i in range(11)])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    lines.append(f'  Strategy diversity (): {entropy:.3f} / {np.log(11):.3f}')
    lines.append(f"{'=' * 70}\n")
    logger.info('\n'.join(lines))

def run_stage2(args, pretrained_checkpoint: str, output_dir: str, logger):
    logger.info('=' * 80)
    logger.info('Stage 2: PPO reinforcement-learning fine-tuning( KL )')
    logger.info('=' * 80)
    if pretrained_checkpoint:
        logger.info(f'Loaded Stage 1 checkpoint: {pretrained_checkpoint}')
    else:
        logger.info('pretraining checkpoint,Starting PPO training')
    logger.info('\nLoading data...')
    train_stocks, val_stocks, test_stocks = load_and_split_data(args, logger)
    train_stocks_serialized = {}
    for code, sdf in train_stocks.items():
        d = {}
        for col in sdf.columns:
            if sdf[col].dtype == 'datetime64[ns]':
                d[col] = sdf[col].dt.strftime('%Y-%m-%d').tolist()
            else:
                d[col] = sdf[col].tolist()
        train_stocks_serialized[code] = d
    reward_config = create_reward_config(args)
    sample_env = create_env(list(train_stocks.values())[0], args, reward_config)
    sample_obs = sample_env.reset()
    obs_shape = {k: v.shape for k, v in sample_obs.items()}
    logger.info(f': {obs_shape}')
    del sample_env
    agent = PPOAgentV2(obs_shape=obs_shape, num_actions=11, lr=args.lr, gamma=args.gamma, lam=args.lam, clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef, value_coef=args.value_coef, max_grad_norm=args.max_grad_norm, n_epochs=args.n_epochs, batch_size=args.batch_size, device=args.device, target_kl=args.target_kl, temperature=args.temperature, model_type=args.model_type, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, kl_coef=args.kl_coef, kl_coef_final=args.kl_coef_final)
    if pretrained_checkpoint:
        agent.load(pretrained_checkpoint)
        agent.load_anchor_policy(pretrained_checkpoint)
        logger.info(f'KL : schedule={args.kl_schedule}, coef={args.kl_coef} → {args.kl_coef_final}, warmup={args.kl_warmup_episodes}')
    else:
        agent.set_kl_coef(0.0)
        logger.info(' PPO : KL (kl_coef=0)')
    total_params = sum((p.numel() for p in agent.model.parameters()))
    logger.info(f'Model parameters: {total_params:,}')
    ckpt_manager = CheckpointManager(output_dir, max_keep=5)
    metrics_tracker = MetricsTracker(output_dir)
    use_parallel = args.num_workers > 0
    sampler = None
    if use_parallel:
        sampler = ParallelSampler(num_workers=args.num_workers, args_dict={'window_size': args.window_size, 'lookback': args.lookback, 'allow_short': getattr(args, 'allow_short', False)}, reward_config=reward_config, logger=logger)
        logger.info(f'Parallel sampling: num_workers={args.num_workers}')
    train_stock_list = list(train_stocks.keys())
    val_stock_list = list(val_stocks.keys())
    test_stock_list = list(test_stocks.keys())
    logger.info(f'\nStarting Stage 2 training ({args.n_episodes} episodes)...')
    best_val_return = -float('inf')
    warmup_episodes = int(args.n_episodes * 0.2)
    validate_kl_schedule_args(args.n_episodes, args.kl_warmup_episodes)
    for episode in range(1, args.n_episodes + 1):
        if episode <= warmup_episodes:
            current_entropy_coef = args.entropy_coef
        else:
            progress = (episode - warmup_episodes) / max(args.n_episodes - warmup_episodes, 1)
            if args.entropy_schedule == 'linear':
                current_entropy_coef = args.entropy_coef - progress * (args.entropy_coef - args.entropy_coef_final)
            else:
                current_entropy_coef = args.entropy_coef_final + 0.5 * (args.entropy_coef - args.entropy_coef_final) * (1 + np.cos(np.pi * progress))
        agent.set_entropy_coef(current_entropy_coef)
        if episode <= warmup_episodes:
            current_temperature = args.temperature
        else:
            progress = (episode - warmup_episodes) / max(args.n_episodes - warmup_episodes, 1)
            if args.entropy_schedule == 'linear':
                current_temperature = args.temperature - progress * (args.temperature - args.temperature_final)
            else:
                current_temperature = args.temperature_final + 0.5 * (args.temperature - args.temperature_final) * (1 + np.cos(np.pi * progress))
        agent.set_temperature(current_temperature)
        if not pretrained_checkpoint:
            current_kl_coef = 0.0
        else:
            current_kl_coef = get_kl_coef(schedule=args.kl_schedule, episode=episode, n_episodes=args.n_episodes, coef_start=args.kl_coef, coef_final=args.kl_coef_final, warmup_episodes=args.kl_warmup_episodes)
        agent.set_kl_coef(current_kl_coef)
        t0 = time.time()
        n_stocks = min(args.stocks_per_episode, len(train_stock_list))
        selected = np.random.choice(train_stock_list, n_stocks, replace=False)
        all_summaries = []
        all_trajectories = []
        if use_parallel and sampler is not None:
            with torch.no_grad():
                episode_cpu_state = {k: v.detach().cpu().numpy() for k, v in agent.model.state_dict().items()}
            episode_model_config = {'model_type': getattr(agent, 'model_type', 'multimodal'), 'num_features': agent.obs_shape['numerical'][1], 'num_actions': agent.num_actions, 'd_model': getattr(agent, 'd_model', 128), 'n_heads': getattr(agent, 'n_heads', 4), 'n_layers': getattr(agent, 'n_layers', 3), 'hidden_dim': 256, 'temperature': getattr(agent, 'temperature', 1.0)}
            num_batches = int(np.ceil(len(selected) / args.num_workers))
            for b in range(num_batches):
                s, e = (b * args.num_workers, min((b + 1) * args.num_workers, len(selected)))
                batch_stocks = selected[s:e]
                stock_df_list = []
                for code in batch_stocks:
                    sdf = train_stocks[code]
                    min_req = args.lookback + args.window_size
                    if len(sdf) >= min_req:
                        stock_df_list.append((code, sdf))
                if not stock_df_list:
                    continue
                batch_summaries, batch_trajectories = parallel_rollout_episode(sampler=sampler, agent=agent, stock_df_list=stock_df_list, rollout_steps=args.rollout_steps, logger=logger, verbose=episode <= 2 and b == 0, pre_cpu_state=episode_cpu_state, pre_model_config=episode_model_config, pre_df_dicts=train_stocks_serialized)
                all_summaries.extend(batch_summaries)
                all_trajectories.extend(batch_trajectories)
        else:
            for code in selected:
                env = create_env(train_stocks[code], args, reward_config)
                summary, trajectory = rollout_episode(env, agent, max_steps=args.rollout_steps, deterministic=False, stock_code=code)
                if len(agent.buffer.dones) > 0:
                    agent.buffer.dones[-1] = True
                all_summaries.append(summary)
                all_trajectories.append(trajectory)
        rollout_time = time.time() - t0
        if use_parallel and len(agent.buffer.dones) > 0:
            offset = 0
            for trajectory in all_trajectories:
                offset += len(trajectory)
                if offset > 0:
                    agent.buffer.dones[offset - 1] = True
        t1 = time.time()
        next_value = 0.0
        update_metrics = agent.update(next_value=next_value)
        update_time = time.time() - t1
        avg_return = np.mean([s.get('avg_return', 0.0) for s in all_summaries])
        avg_sharpe = np.mean([s.get('avg_sharpe', 0.0) for s in all_summaries])
        avg_reward = np.mean([s.get('avg_reward', 0.0) for s in all_summaries])
        train_metrics = {'episode': episode, 'avg_return': avg_return, 'avg_sharpe': avg_sharpe, 'avg_reward': avg_reward, 'entropy_coef': current_entropy_coef, 'temperature': current_temperature, 'kl_coef': current_kl_coef, **update_metrics}
        metrics_tracker.log_metrics(train_metrics, 'train', episode)
        if episode % 10 == 0 or episode == 1:
            logger.info(f"Episode {episode}/{args.n_episodes} | Return: {avg_return:.4f} | Sharpe: {avg_sharpe:.3f} | Reward: {avg_reward:.4f} | PolicyLoss: {update_metrics.get('policy_loss', 0):.4f} | KL_anchor: {update_metrics.get('kl_anchor_loss', 0):.4f} (coef={current_kl_coef:.4f}) | Entropy: {update_metrics.get('entropy', 0):.4f} (coef={current_entropy_coef:.4f}) | Temp: {current_temperature:.2f} | Time: rollout={rollout_time:.1f}s, update={update_time:.1f}s")
            all_actions = [step['action'] for traj in all_trajectories for step in traj]
            print_action_dist(all_actions, episode, logger)
        if episode % args.eval_interval == 0:
            logger.info(f"\n{'-' * 80}")
            logger.info(f'validation (Episode {episode})...')
            n_val = min(3, len(val_stock_list))
            val_stocks_sample = np.random.choice(val_stock_list, n_val, replace=False)
            all_val_metrics_greedy = []
            all_val_metrics_topk = []
            for code in val_stocks_sample:
                val_env = create_env(val_stocks[code], args, reward_config, sequential_eval=True)
                stock_metrics = evaluate(val_env, agent, n_episodes=1, eval_mode=args.eval_mode)
                if args.eval_mode == 'both':
                    all_val_metrics_greedy.append(stock_metrics['greedy'])
                    all_val_metrics_topk.append(stock_metrics['topk'])
                elif args.eval_mode == 'greedy':
                    all_val_metrics_greedy.append(stock_metrics)
                else:
                    all_val_metrics_topk.append(stock_metrics)
            logger.info(f'validation:')
            if args.eval_mode in ('both', 'greedy') and all_val_metrics_greedy:
                val_metrics_greedy = {'avg_return': np.mean([m['avg_return'] for m in all_val_metrics_greedy]), 'std_return': np.std([m['avg_return'] for m in all_val_metrics_greedy]), 'avg_sharpe': np.mean([m['avg_sharpe'] for m in all_val_metrics_greedy]), 'avg_trades': np.mean([m['avg_trades'] for m in all_val_metrics_greedy]), 'avg_reward': np.mean([m['avg_reward'] for m in all_val_metrics_greedy]), 'action_entropy': np.mean([m['action_entropy'] for m in all_val_metrics_greedy]), 'strategy_distribution': all_val_metrics_greedy[0]['strategy_distribution']}
                metrics_tracker.log_metrics(val_metrics_greedy, 'val_greedy', episode)
                logger.info(f'\n  [Greedy ()]')
                logger.info(f"    Average return: {val_metrics_greedy['avg_return']:.4f} ± {val_metrics_greedy['std_return']:.4f}")
                logger.info(f"    Average Sharpe: {val_metrics_greedy['avg_sharpe']:.3f}")
                logger.info(f"    Average Reward: {val_metrics_greedy['avg_reward']:.4f}")
                logger.info(f"    Action entropy: {val_metrics_greedy['action_entropy']:.3f}")
                logger.info(f'    Strategy distribution:')
                for i in range(11):
                    ratio = val_metrics_greedy['strategy_distribution'].get(f'strategy_{i}', 0.0)
                    if ratio > 0.01:
                        logger.info(f"      [{i}] {STRATEGY_ACTIONS[i]['name']}: {ratio * 100:.1f}%")
            if args.eval_mode in ('both', 'topk') and all_val_metrics_topk:
                val_metrics_topk = {'avg_return': np.mean([m['avg_return'] for m in all_val_metrics_topk]), 'std_return': np.std([m['avg_return'] for m in all_val_metrics_topk]), 'avg_sharpe': np.mean([m['avg_sharpe'] for m in all_val_metrics_topk]), 'avg_trades': np.mean([m['avg_trades'] for m in all_val_metrics_topk]), 'avg_reward': np.mean([m['avg_reward'] for m in all_val_metrics_topk]), 'action_entropy': np.mean([m['action_entropy'] for m in all_val_metrics_topk]), 'strategy_distribution': all_val_metrics_topk[0]['strategy_distribution']}
                metrics_tracker.log_metrics(val_metrics_topk, 'val_topk', episode)
                logger.info(f'\n  [Top-3 ]')
                logger.info(f"    Average return: {val_metrics_topk['avg_return']:.4f} ± {val_metrics_topk['std_return']:.4f}")
                logger.info(f"    Average Sharpe: {val_metrics_topk['avg_sharpe']:.3f}")
                logger.info(f"    Average Reward: {val_metrics_topk['avg_reward']:.4f}")
                logger.info(f"    Action entropy: {val_metrics_topk['action_entropy']:.3f} (=2.398)")
                logger.info(f'    Strategy distribution:')
                for i in range(11):
                    ratio = val_metrics_topk['strategy_distribution'].get(f'strategy_{i}', 0.0)
                    if ratio > 0.01:
                        logger.info(f"      [{i}] {STRATEGY_ACTIONS[i]['name']}: {ratio * 100:.1f}%")
            if args.eval_mode in ('greedy', 'both') and all_val_metrics_greedy:
                vr = val_metrics_greedy['avg_return']
            elif all_val_metrics_topk:
                vr = val_metrics_topk['avg_return']
            else:
                vr = -float('inf')
            if vr > best_val_return:
                best_val_return = vr
                logger.info(f'\n✓ New best model! val_return={best_val_return:.4f}')
                state = {'episode': episode, 'model_state_dict': agent.model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'best_val_return': best_val_return, 'args': vars(args), 'stage': 'rl_finetune'}
                ckpt_manager.save_checkpoint(state, is_best=True, filename='best_model.pt')
            logger.info(f"{'-' * 80}")
        if episode % args.save_interval == 0:
            state = {'episode': episode, 'model_state_dict': agent.model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'args': vars(args)}
            ckpt_manager.save_checkpoint(state, is_best=False, filename=f'checkpoint_ep{episode}.pt')
    if sampler is not None:
        sampler.close()
    state = {'episode': args.n_episodes, 'model_state_dict': agent.model.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(), 'args': vars(args)}
    ckpt_manager.save_checkpoint(state, is_best=False, filename='last_model.pt')
    logger.info('\n' + '=' * 80)
    logger.info('test...')
    logger.info('=' * 80)
    try:
        best_ckpt = ckpt_manager.load_checkpoint('best_model.pt')
        agent.model.load_state_dict(best_ckpt['model_state_dict'])
        logger.info('✓ Loadedvalidation')
    except FileNotFoundError:
        logger.warning('⚠️  best_model.pt,Final model')
    logger.info(f' {len(test_stock_list)} test...')
    all_test_metrics_greedy = []
    all_test_metrics_topk = []
    for code in test_stock_list:
        test_env = create_env(test_stocks[code], args, reward_config, sequential_eval=True)
        stock_metrics = evaluate(test_env, agent, n_episodes=1, eval_mode='both')
        all_test_metrics_greedy.append(stock_metrics['greedy'])
        all_test_metrics_topk.append(stock_metrics['topk'])
        logger.info(f"  {code}: greedy_return={stock_metrics['greedy']['avg_return']:.4f}, greedy_mdd={stock_metrics['greedy']['avg_max_drawdown']:.4f}, topk_return={stock_metrics['topk']['avg_return']:.4f}, topk_mdd={stock_metrics['topk']['avg_max_drawdown']:.4f}")
    test_metrics_greedy = {'avg_return': np.mean([m['avg_return'] for m in all_test_metrics_greedy]), 'std_return': np.std([m['avg_return'] for m in all_test_metrics_greedy]), 'avg_sharpe': np.mean([m['avg_sharpe'] for m in all_test_metrics_greedy]), 'avg_trades': np.mean([m['avg_trades'] for m in all_test_metrics_greedy]), 'avg_reward': np.mean([m['avg_reward'] for m in all_test_metrics_greedy]), 'avg_max_drawdown': np.mean([m['avg_max_drawdown'] for m in all_test_metrics_greedy]), 'action_entropy': np.mean([m['action_entropy'] for m in all_test_metrics_greedy]), 'strategy_distribution': all_test_metrics_greedy[0]['strategy_distribution']}
    test_metrics_topk = {'avg_return': np.mean([m['avg_return'] for m in all_test_metrics_topk]), 'std_return': np.std([m['avg_return'] for m in all_test_metrics_topk]), 'avg_sharpe': np.mean([m['avg_sharpe'] for m in all_test_metrics_topk]), 'avg_trades': np.mean([m['avg_trades'] for m in all_test_metrics_topk]), 'avg_reward': np.mean([m['avg_reward'] for m in all_test_metrics_topk]), 'avg_max_drawdown': np.mean([m['avg_max_drawdown'] for m in all_test_metrics_topk]), 'action_entropy': np.mean([m['action_entropy'] for m in all_test_metrics_topk]), 'strategy_distribution': all_test_metrics_topk[0]['strategy_distribution']}
    metrics_tracker.log_metrics(test_metrics_greedy, 'test_greedy', args.n_episodes)
    metrics_tracker.log_metrics(test_metrics_topk, 'test_topk', args.n_episodes)
    logger.info(f'\ntest (Average {len(test_stock_list)} ):')
    logger.info(f'\n  [Greedy ()]')
    logger.info(f"    Average return: {test_metrics_greedy['avg_return']:.4f} ± {test_metrics_greedy['std_return']:.4f}")
    logger.info(f"    Average Sharpe: {test_metrics_greedy['avg_sharpe']:.3f}")
    logger.info(f"    Average maximum drawdown: {test_metrics_greedy['avg_max_drawdown']:.4f} ({test_metrics_greedy['avg_max_drawdown'] * 100:.2f}%)")
    logger.info(f"    Average trades: {test_metrics_greedy['avg_trades']:.1f}")
    logger.info(f"    Action entropy: {test_metrics_greedy['action_entropy']:.3f}")
    logger.info(f'    Strategy distribution:')
    for i in range(11):
        ratio = test_metrics_greedy['strategy_distribution'].get(f'strategy_{i}', 0.0)
        if ratio > 0.01:
            logger.info(f"      [{i}] {STRATEGY_ACTIONS[i]['name']}: {ratio * 100:.1f}%")
    logger.info(f'\n  [Top-3 ]')
    logger.info(f"    Average return: {test_metrics_topk['avg_return']:.4f} ± {test_metrics_topk['std_return']:.4f}")
    logger.info(f"    Average Sharpe: {test_metrics_topk['avg_sharpe']:.3f}")
    logger.info(f"    Average maximum drawdown: {test_metrics_topk['avg_max_drawdown']:.4f} ({test_metrics_topk['avg_max_drawdown'] * 100:.2f}%)")
    logger.info(f"    Average trades: {test_metrics_topk['avg_trades']:.1f}")
    logger.info(f"    Action entropy: {test_metrics_topk['action_entropy']:.3f} (=2.398)")
    logger.info(f'    Strategy distribution:')
    for i in range(11):
        ratio = test_metrics_topk['strategy_distribution'].get(f'strategy_{i}', 0.0)
        if ratio > 0.01:
            logger.info(f"      [{i}] {STRATEGY_ACTIONS[i]['name']}: {ratio * 100:.1f}%")
    metrics_tracker.save_metrics()
    logger.info('=' * 80)
    logger.info('Stage 2 completed!')
    logger.info(f'Output directory: {output_dir}')
    logger.info('=' * 80)

def main():
    args = parse_args()
    set_seed(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logger(run_dir, name='two_stage_train')
    logger.info('Two-stage training started')
    logger.info(f'Run directory: {run_dir}')
    pretrained_ckpt = args.pretrained_checkpoint
    if not args.skip_stage1:
        logger.info('\n' + '=' * 80)
        logger.info('>>> Starting Stage 1: behavior cloningpretraining')
        logger.info('=' * 80)
        s1_args = argparse.Namespace(data_path=args.data_path, train_start=args.train_start, train_end=args.train_end, val_start=args.val_start, val_end=args.val_end, lookback=args.lookback, model_type=args.model_type, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, dropout=0.1, n_epochs=args.s1_n_epochs, batch_size=args.s1_batch_size, lr=args.s1_lr, weight_decay=args.s1_weight_decay, max_grad_norm=args.s1_max_grad_norm, warmup_ratio=args.s1_warmup_ratio, entropy_coef=args.s1_entropy_coef, unfreeze_value_head_epoch=args.s1_unfreeze_value_head_epoch, value_loss_coef=getattr(args, 's1_value_loss_coef', 0.5), value_head_lr_ratio=getattr(args, 's1_value_head_lr_ratio', 0.1), seed=args.seed, device=args.device, num_workers=args.s1_num_workers, log_interval=50, output_dir=run_dir)
        pretrained_ckpt = run_pretrain(s1_args)
        logger.info(f'\n>>> Stage 1 completed!checkpoint: {pretrained_ckpt}\n')
    elif pretrained_ckpt is None:
        logger.info('\n>>>  Stage 1,Starting Stage 2( PPO training)\n')
    else:
        logger.info(f'\n>>>  Stage 1, checkpoint: {pretrained_ckpt}\n')
    if args.stage1_only:
        logger.info('--stage1_only , Stage 2.')
        return
    logger.info('=' * 80)
    logger.info('>>> Starting Stage 2: PPO reinforcement-learning fine-tuning')
    logger.info('=' * 80)
    s2_dir = os.path.join(run_dir, 'stage2_rl')
    os.makedirs(s2_dir, exist_ok=True)
    run_stage2(args, pretrained_ckpt, s2_dir, logger)
    logger.info('\nTwo-stage training completed!')
    logger.info(f'Output directory: {run_dir}')
if __name__ == '__main__':
    main()
