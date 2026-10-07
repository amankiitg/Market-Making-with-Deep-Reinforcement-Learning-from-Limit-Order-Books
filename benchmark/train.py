"""Training entry point for the RL agent.

The agent is deliberately kept plain: a stock PPO with a small MLP, the standard
hyperparameters, one environment per process, and a fixed step budget per seed. Nothing is
tuned per setting and no setting-specific feature is provided, so the same recipe is applied
to both markets. What makes the task learnable within a few minutes of CPU is the design of
the problem, not the algorithm: a two dimensional observation, an action range of a few times
the optimal depth, a reward scaled by the known optimum, and the mean-zero marking noise
removed (see benchmark/envs.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmark.config import Setting


@dataclass
class TrainingConfig:
    total_timesteps: int = 500_000
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    net_arch: tuple = (64, 64)
    log_std_init: float = -1.0
    noise_free: bool = True
    eval_every: int = 25_000


@dataclass
class TrainingRun:
    seed: int
    model: object
    history: list = field(default_factory=list)   # (timesteps, mean objective on the eval set)
    wall_time: float = 0.0
    config: TrainingConfig | None = None


def train_ppo(setting: Setting, seed: int, j_opt: float, eval_crn=None,
              config: TrainingConfig | None = None):
    """Train one PPO agent and return a TrainingRun.

    j_opt is used only to scale the reward, which leaves the optimal policy unchanged: every
    policy is a maximiser of any positive multiple of the objective.
    """
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from benchmark.envs import MMBenchEnv
    from benchmark.simulator import simulate
    from benchmark.strategies import RLPolicy

    config = config or TrainingConfig()
    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 1))))

    scale = 1.0 / abs(j_opt)
    vec_env = DummyVecEnv([lambda: MMBenchEnv(setting, seed=seed,
                                              noise_free=config.noise_free,
                                              reward_scale=scale)])
    model = PPO('MlpPolicy', vec_env, seed=seed, n_steps=config.n_steps,
                batch_size=config.batch_size, n_epochs=config.n_epochs,
                learning_rate=config.learning_rate, gamma=config.gamma,
                gae_lambda=config.gae_lambda, ent_coef=config.ent_coef,
                policy_kwargs=dict(net_arch=list(config.net_arch),
                                   log_std_init=config.log_std_init),
                device='cpu', verbose=0)

    class _Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = []

        def _on_step(self) -> bool:
            if eval_crn is not None and self.num_timesteps % config.eval_every == 0:
                rollout = simulate(setting, RLPolicy(self.model, setting), eval_crn)
                self.history.append((int(self.num_timesteps), float(rollout.pnl.mean())))
            return True

    callback = _Eval()
    start = time.time()
    model.learn(total_timesteps=config.total_timesteps, callback=callback,
                progress_bar=False)
    wall_time = time.time() - start
    if eval_crn is not None and not callback.history:
        rollout = simulate(setting, RLPolicy(model, setting), eval_crn)
        callback.history.append((int(config.total_timesteps), float(rollout.pnl.mean())))
    return TrainingRun(seed=seed, model=model, history=callback.history, wall_time=wall_time,
                       config=config)
