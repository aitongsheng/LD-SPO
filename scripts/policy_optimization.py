import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from ldspo_model import LDSPOMultimodalStudent

class RolloutBuffer:

    def __init__(self, buffer_size: int, obs_shape: Dict, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.observations: List[Dict] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.ptr = 0
        self.path_start_idx = 0

    def add(self, obs: Dict[str, np.ndarray], action: int, reward: float, value: float, log_prob: float, done: bool):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1

    def get(self) -> Optional[Dict]:
        if len(self.observations) == 0:
            return None
        return {'observations': self.observations, 'actions': np.array(self.actions, dtype=np.int64), 'rewards': np.array(self.rewards, dtype=np.float32), 'values': np.array(self.values, dtype=np.float32), 'log_probs': np.array(self.log_probs, dtype=np.float32), 'dones': np.array(self.dones, dtype=np.float32)}

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.ptr = 0
        self.path_start_idx = 0

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float, gamma: float=0.99, lam: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    returns = advantages + values
    return (advantages, returns)

class PPOAgentV2:

    def __init__(self, obs_shape: Dict, num_actions: int=11, lr: float=0.0001, gamma: float=0.95, lam: float=0.5, clip_epsilon: float=0.2, entropy_coef: float=0.01, value_coef: float=2.0, max_grad_norm: float=0.5, n_epochs: int=10, batch_size: int=128, device: str='cuda', target_kl: float=0.15, temperature: float=1.2, model_type: str='multimodal', d_model: int=128, n_heads: int=4, n_layers: int=3, kl_coef: float=0.1, kl_coef_final: float=0.0):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.target_kl = target_kl
        self.temperature = temperature
        self.model_type = model_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kl_coef = kl_coef
        self.kl_coef_final = kl_coef_final
        self.anchor_model: Optional[nn.Module] = None
        numerical_dim = obs_shape['numerical'][1]
        seq_len = obs_shape['numerical'][0]
        self.model = self._build_model(model_type, numerical_dim, seq_len, d_model, n_heads, n_layers, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer(buffer_size=10000, obs_shape=obs_shape, device=self.device)

    def _build_model(self, model_type, numerical_dim, seq_len, d_model, n_heads, n_layers, num_actions) -> nn.Module:
        if model_type != 'multimodal':
            raise ValueError('Only the paper-aligned multimodal model is included')
        return LDSPOMultimodalStudent(numerical_dim=numerical_dim, seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers, num_actions=num_actions, dropout=0.1)

    def load_anchor_policy(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.anchor_model = copy.deepcopy(self.model)
        model_dict = self.anchor_model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        self.anchor_model.load_state_dict(model_dict, strict=False)
        self.anchor_model.to(self.device)
        for param in self.anchor_model.parameters():
            param.requires_grad = False
        self.anchor_model.eval()
        n_loaded = len(filtered)
        n_total = len(model_dict)
        print(f'✓ KL Anchor Policy Loaded: {n_loaded}/{n_total}  ({checkpoint_path})')

    def set_kl_coef(self, coef: float):
        self.kl_coef = coef

    def set_entropy_coef(self, coef: float):
        self.entropy_coef = coef

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def select_action(self, obs: Dict[str, np.ndarray], deterministic: bool=False, debug: bool=False) -> Tuple[int, float, float]:
        self.model.eval()
        with torch.no_grad():
            model_input = self._single_obs_to_model_input(obs)
            action_t, log_prob_t, value_t, entropy_t = self.model.get_action(model_input, deterministic=deterministic, temperature=self.temperature)
            action = action_t.item()
            log_prob = log_prob_t.item()
            value = value_t.item()
            if debug:
                logits, _ = self.model(model_input)
                probs = torch.softmax(logits / self.temperature, dim=-1)
                print(f'  [Debug] Action: {action}, Value: {value:.4f}, Entropy: {entropy_t.item():.4f}')
                print(f'  [Debug] Probs: {probs.cpu().numpy().round(3)}')
        return (action, log_prob, value)

    def policy_distribution(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        self.model.eval()
        with torch.no_grad():
            model_input = self._single_obs_to_model_input(obs)
            logits, value = self.model(model_input)
            probs = torch.softmax(logits / self.temperature, dim=-1)
        return (probs.squeeze(0).cpu().numpy(), float(value.squeeze().item()))

    def update(self, next_value: float=0.0) -> Dict[str, float]:
        data = self.buffer.get()
        if data is None:
            return {}
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        values = data['values']
        old_log_probs = data['log_probs']
        dones = data['dones']
        advantages, returns = compute_gae(rewards, values, dones, next_value, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_anchor = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        early_stopped = False
        early_stop_epoch = self.n_epochs
        early_stop_kl = 0.0
        self.model.train()
        anchor_log_probs_all: Optional[torch.Tensor] = None
        if self.anchor_model is not None and self.kl_coef > 0:
            _chunk = 512
            _anchor_chunks = []
            with torch.no_grad():
                for _s in range(0, len(observations), _chunk):
                    _batch_obs = [observations[i] for i in range(_s, min(_s + _chunk, len(observations)))]
                    _batch_input = self._batch_obs_to_tensor(_batch_obs)
                    _logits, _ = self.anchor_model(_batch_input)
                    _anchor_chunks.append(F.log_softmax(_logits, dim=-1))
            anchor_log_probs_all = torch.cat(_anchor_chunks, dim=0)
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(observations))
            epoch_kl = 0.0
            epoch_updates = 0
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_obs = [observations[i] for i in batch_indices]
                batch_input = self._batch_obs_to_tensor(batch_obs)
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                current_logits, new_values = self.model(batch_input)
                new_values = new_values.squeeze(-1)
                dist = torch.distributions.Categorical(logits=current_logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                kl_anchor_loss = torch.tensor(0.0, device=self.device)
                if anchor_log_probs_all is not None:
                    batch_anchor_log_probs = anchor_log_probs_all[batch_indices]
                    current_log_probs = F.log_softmax(current_logits, dim=-1)
                    current_probs = current_log_probs.exp()
                    kl_anchor_loss = F.kl_div(batch_anchor_log_probs, current_probs, reduction='batchmean', log_target=False)
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + self.kl_coef * kl_anchor_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl_anchor += kl_anchor_loss.item()
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    total_approx_kl += approx_kl
                    epoch_kl += abs(approx_kl)
                    epoch_updates += 1
                n_updates += 1
            avg_epoch_kl = epoch_kl / max(epoch_updates, 1)
            if avg_epoch_kl > self.target_kl:
                early_stopped = True
                early_stop_epoch = epoch + 1
                early_stop_kl = avg_epoch_kl
                break
        self.buffer.clear()
        with torch.no_grad():
            value_mean = float(np.mean(values))
            value_std = float(np.std(values))
            value_nonzero = float(np.mean(np.abs(values) > 1e-06))
            if len(returns) > 1 and np.std(returns) > 1e-08 and (np.std(values) > 1e-08):
                value_return_corr = float(np.corrcoef(values, returns)[0, 1])
            else:
                value_return_corr = 0.0
        metrics = {'policy_loss': total_policy_loss / max(n_updates, 1), 'value_loss': total_value_loss / max(n_updates, 1), 'entropy': total_entropy / max(n_updates, 1), 'approx_kl': total_approx_kl / max(n_updates, 1), 'kl_anchor_loss': total_kl_anchor / max(n_updates, 1), 'kl_coef': self.kl_coef, 'explained_variance': self._explained_variance(values, returns), 'n_updates': n_updates, 'n_epochs_total': self.n_epochs, 'early_stop_epoch': early_stop_epoch if early_stopped else self.n_epochs, 'early_stopped': early_stopped, 'early_stop_kl': early_stop_kl, 'value_mean': value_mean, 'value_std': value_std, 'value_nonzero_ratio': value_nonzero, 'value_return_corr': value_return_corr, 'reward_mean': float(np.mean(rewards)), 'reward_std': float(np.std(rewards)), 'reward_min': float(np.min(rewards)), 'reward_max': float(np.max(rewards)), 'reward_positive_ratio': float(np.mean(rewards > 0))}
        return metrics

    def _single_obs_to_model_input(self, obs: Dict[str, np.ndarray]):
        numerical = torch.tensor(obs['numerical'], dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.model_type != 'multimodal':
            return numerical
        return {'numerical': numerical, 'news_token_ids': torch.tensor(obs['news_token_ids'], dtype=torch.long, device=self.device).unsqueeze(0), 'news_mask': torch.tensor(obs['news_mask'], dtype=torch.bool, device=self.device).unsqueeze(0)}

    def _batch_obs_to_tensor(self, obs_list: List[Dict]):
        numerical = torch.tensor(np.stack([obs['numerical'] for obs in obs_list]), dtype=torch.float32, device=self.device)
        if self.model_type != 'multimodal':
            return numerical
        return {'numerical': numerical, 'news_token_ids': torch.tensor(np.stack([obs['news_token_ids'] for obs in obs_list]), dtype=torch.long, device=self.device), 'news_mask': torch.tensor(np.stack([obs['news_mask'] for obs in obs_list]), dtype=torch.bool, device=self.device)}

    def _explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        var_y = np.var(y_true)
        if var_y < 1e-08:
            return 0.0
        return 1 - np.var(y_true - y_pred) / var_y

    def save(self, filepath: str):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, filepath)

    def load(self, filepath: str, freeze_encoder: bool=False):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'feature_extractor' in name or 'input_projection' in name:
                    param.requires_grad = False
        print(f'✓ Loaded: {len(pretrained_dict)}/{len(model_dict)}  ({filepath})')

    def set_reward_normalization(self, enable: bool, momentum: float=0.99):
        pass
