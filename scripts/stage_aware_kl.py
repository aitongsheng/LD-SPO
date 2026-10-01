import math
KL_SCHEDULES = ('none', 'constant', 'linear', 'cosine')

def get_kl_coef(schedule: str, episode: int, n_episodes: int, coef_start: float, coef_final: float, warmup_episodes: int=0) -> float:
    if schedule not in KL_SCHEDULES:
        raise ValueError(f'Unsupported KL schedule: {schedule}')
    if episode < 1 or n_episodes < 1:
        raise ValueError('episode and n_episodes must be positive')
    if not 0 <= warmup_episodes <= n_episodes:
        raise ValueError('kl_warmup_episodes must be in [0, n_episodes]')
    if coef_start < 0 or coef_final < 0:
        raise ValueError('KL coefficients must be non-negative')
    if schedule == 'none':
        return 0.0
    if schedule == 'constant':
        return coef_start
    if episode <= warmup_episodes:
        return coef_start
    decay_episodes = n_episodes - warmup_episodes
    if decay_episodes <= 1:
        progress = 1.0
    else:
        progress = (episode - warmup_episodes - 1) / (decay_episodes - 1)
        progress = min(max(progress, 0.0), 1.0)
    if schedule == 'linear':
        weight = 1.0 - progress
    else:
        weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return coef_final + (coef_start - coef_final) * weight

def validate_kl_schedule_args(n_episodes: int, warmup_episodes: int) -> None:
    if not 0 <= warmup_episodes <= n_episodes:
        raise ValueError('kl_warmup_episodes must be in [0, n_episodes]')
