import os
_BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from typing import Dict, Optional
from teacher_distillation_data import BCDataset, build_dataloader, TEACHER_COLS, NUM_ACTIONS
from ldspo_model import LDSPOMultimodalStudent
from ldspo_config import STRATEGY_ACTIONS, NUMERICAL_FEATURE_COLS
from experiment_utils import set_seed, setup_logger
from market_data import load_table

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

def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()

def kl_divergence_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs_model = F.log_softmax(logits, dim=-1)
    kl = F.kl_div(log_probs_model, soft_targets, reduction='batchmean')
    return kl

@torch.no_grad()
def evaluate_bc(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_ce = 0.0
    total_kl = 0.0
    correct_top1 = 0
    correct_top3 = 0
    n_samples = 0
    for batch in val_loader:
        obs = {k: v.to(device) for k, v in batch[0].items()}
        teacher = batch[1].to(device)
        logits, _ = model(obs)
        ce = soft_cross_entropy(logits, teacher)
        kl = kl_divergence_loss(logits, teacher)
        teacher_label = teacher.argmax(dim=-1)
        model_pred = logits.argmax(dim=-1)
        correct_top1 += (model_pred == teacher_label).sum().item()
        top3_preds = logits.topk(3, dim=-1).indices
        for i in range(teacher_label.shape[0]):
            if teacher_label[i].item() in top3_preds[i].tolist():
                correct_top3 += 1
        batch_size = teacher.shape[0]
        total_ce += ce.item() * batch_size
        total_kl += kl.item() * batch_size
        n_samples += batch_size
    return {'ce_loss': total_ce / max(n_samples, 1), 'kl_loss': total_kl / max(n_samples, 1), 'top1_acc': correct_top1 / max(n_samples, 1), 'top3_acc': correct_top3 / max(n_samples, 1)}

def pretrain(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    action_indices: Optional[list] = getattr(args, 'action_indices', None)
    effective_num_actions = len(action_indices) if action_indices is not None else NUM_ACTIONS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'pretrain_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir, name='pretrain')
    logger.info('=' * 80)
    logger.info('Stage 1: behavior cloning(Behavior Cloning)pretraining')
    logger.info('=' * 80)
    logger.info(f'Device: {device}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Action space: {effective_num_actions} ' + (f' (: {action_indices})' if action_indices else ' ( 11 )'))
    for k, v in vars(args).items():
        logger.info(f'  {k}: {v}')
    logger.info('\nLoading data...')
    df = load_table(args.data_path)
    logger.info(f'  : {df.shape}')
    df = _patch_df_columns(df)
    logger.info(f'  : {df.shape}, ={len(df.columns)}')
    train_dataset = BCDataset(df=df, lookback=args.lookback, date_start=args.train_start, date_end=args.train_end, verbose=True, action_indices=action_indices)
    val_dataset = BCDataset(df=df, lookback=args.lookback, date_start=args.val_start, date_end=args.val_end, verbose=True, action_indices=action_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    logger.info(f'\ntraining: {len(train_dataset)} , {len(train_loader)} ')
    logger.info(f'validation: {len(val_dataset)} , {len(val_loader)} ')
    sample_obs = train_dataset[0][0]
    seq_len, num_features = sample_obs['numerical'].shape
    logger.info(f"\n: numerical=({seq_len}, {num_features}), news_token_ids={tuple(sample_obs['news_token_ids'].shape)}")
    if args.model_type != 'multimodal':
        raise ValueError('Only the paper-aligned multimodal model is included')
    model = LDSPOMultimodalStudent(numerical_dim=num_features, seq_len=seq_len, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, num_actions=effective_num_actions, dropout=args.dropout).to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(f': {args.model_type}, : {total_params:,} (training: {trainable_params:,})')
    for name, param in model.named_parameters():
        if 'value_head' in name:
            param.requires_grad = False
    trainable_after_freeze = sum((p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(f' Value Head training: {trainable_after_freeze:,}')
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.n_epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info('\nvalidation(training)...')
    val_metrics = evaluate_bc(model, val_loader, device)
    logger.info(f"   CE Loss: {val_metrics['ce_loss']:.4f} | KL: {val_metrics['kl_loss']:.4f} | Top-1 Acc: {val_metrics['top1_acc'] * 100:.2f}% | Top-3 Acc: {val_metrics['top3_acc'] * 100:.2f}%")
    logger.info('\nStartingtraining...')
    logger.info(f"  Value Head  epoch: {args.unfreeze_value_head_epoch} ({('' if args.unfreeze_value_head_epoch <= 0 else '')})")
    if train_dataset.has_value_target:
        logger.info(f'  Value Loss : {args.value_loss_coef}( teacher )')
    else:
        logger.info(f'  Value Loss: ( teacher_value_target , compute_value_targets.py)')
    best_val_kl = float('inf')
    best_epoch = 0
    global_step = 0
    value_head_active = False
    train_history = []
    val_history = []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_ce = 0.0
        epoch_kl = 0.0
        epoch_entropy = 0.0
        epoch_value_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch_idx, (obs, teacher, value_target) in enumerate(train_loader):
            obs = {k: v.to(device) for k, v in obs.items()}
            teacher = teacher.to(device)
            value_target = value_target.to(device)
            logits, value = model(obs)
            ce_loss = soft_cross_entropy(logits, teacher)
            if args.entropy_coef > 0:
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = ce_loss - args.entropy_coef * entropy
            else:
                loss = ce_loss
                entropy = torch.tensor(0.0)
            value_loss = torch.tensor(0.0, device=device)
            if value_head_active and train_dataset.has_value_target:
                value_loss = F.mse_loss(value.squeeze(-1), value_target)
                loss = loss + args.value_loss_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                kl = kl_divergence_loss(logits, teacher)
            epoch_ce += ce_loss.item()
            epoch_kl += kl.item()
            epoch_entropy += entropy.item()
            epoch_value_loss += value_loss.item()
            n_batches += 1
            global_step += 1
            if (batch_idx + 1) % args.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                vl_str = f' | VLoss: {value_loss.item():.4f}' if value_head_active else ''
                logger.info(f'  Epoch {epoch}/{args.n_epochs} | Step {batch_idx + 1}/{len(train_loader)} | CE: {ce_loss.item():.4f} | KL: {kl.item():.4f} | Entropy: {entropy.item():.4f}{vl_str} | LR: {current_lr:.2e}')
        train_ce = epoch_ce / max(n_batches, 1)
        train_kl = epoch_kl / max(n_batches, 1)
        train_entropy = epoch_entropy / max(n_batches, 1)
        train_value_loss = epoch_value_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        val_metrics = evaluate_bc(model, val_loader, device)
        train_history.append({'epoch': epoch, 'ce': train_ce, 'kl': train_kl, 'entropy': train_entropy, 'value_loss': train_value_loss})
        val_history.append({'epoch': epoch, **val_metrics})
        vl_str = f', VLoss={train_value_loss:.4f}' if value_head_active else ''
        logger.info(f"\nEpoch {epoch}/{args.n_epochs} | Time: {elapsed:.1f}s\n  Train: CE={train_ce:.4f}, KL={train_kl:.4f}, Entropy={train_entropy:.4f}{vl_str}\n  Val:   CE={val_metrics['ce_loss']:.4f}, KL={val_metrics['kl_loss']:.4f}, Top-1={val_metrics['top1_acc'] * 100:.2f}%, Top-3={val_metrics['top3_acc'] * 100:.2f}%")
        if val_metrics['kl_loss'] < best_val_kl:
            best_val_kl = val_metrics['kl_loss']
            best_epoch = epoch
            best_ckpt_path = os.path.join(output_dir, 'best_pretrain.pt')
            _save_checkpoint(model, optimizer, epoch, val_metrics, args, best_ckpt_path)
            logger.info(f'  ✓ New best model! Val KL: {best_val_kl:.4f} → : {best_ckpt_path}')
        if args.unfreeze_value_head_epoch > 0 and epoch == args.unfreeze_value_head_epoch:
            value_params = []
            for name, param in model.named_parameters():
                if 'value_head' in name:
                    param.requires_grad = True
                    value_params.append(param)
            if value_params:
                optimizer.add_param_group({'params': value_params, 'lr': args.lr * args.value_head_lr_ratio, 'weight_decay': args.weight_decay})
                value_head_active = True
                logger.info(f'  ✓ Epoch {epoch}:  Value Head ({len(value_params)} ), LR={args.lr * args.value_head_lr_ratio:.2e}, Value Loss ={args.value_loss_coef}')
    final_ckpt_path = os.path.join(output_dir, 'final_pretrain.pt')
    _save_checkpoint(model, optimizer, args.n_epochs, val_metrics, args, final_ckpt_path)
    logger.info(f'\nFinal model: {final_ckpt_path}')
    logger.info('=' * 80)
    logger.info('Stage 1 pretrainingcompleted!')
    logger.info(f'   Epoch: {best_epoch},  Val KL: {best_val_kl:.4f}')
    logger.info(f'  Best model: {best_ckpt_path}')
    logger.info(f'  Final model: {final_ckpt_path}')
    logger.info('=' * 80)
    return best_ckpt_path

def _save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_metrics: Dict, args: argparse.Namespace, filepath: str):
    full_state_dict = {}
    for name, param in model.named_parameters():
        full_state_dict[name] = param.data
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_metrics': val_metrics, 'args': vars(args), 'stage': 'pretrain'}, filepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join(_BUNDLE_ROOT, 'data', 'us_stocks_ohlcv_with_features_and_teachers.parquet'))
    parser.add_argument('--train_start', type=str, default='2018-01-01')
    parser.add_argument('--train_end', type=str, default='2020-12-31')
    parser.add_argument('--val_start', type=str, default='2021-01-01')
    parser.add_argument('--val_end', type=str, default='2022-12-31')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--model_type', type=str, default='multimodal', choices=['multimodal'])
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--unfreeze_value_head_epoch', type=int, default=20)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--value_head_lr_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default=os.path.join(_BUNDLE_ROOT, 'outputs', 'pretrain'))
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    best_ckpt = pretrain(args)
    print(f'\n✓ pretrainingcompleted, checkpoint: {best_ckpt}')
