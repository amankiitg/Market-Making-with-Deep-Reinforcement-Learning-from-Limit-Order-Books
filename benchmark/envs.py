"""Gymnasium wrapper around the discrete simulator, for the RL agent.

One episode is one trading day. The observation is (t / T, q / Q), which is deliberately the
minimal information that matters: the optimal depth in both settings is a function of time
and inventory only, exactly as the HJB solution and the dynamic program both show. A learning
failure therefore cannot be blamed on a missing feature, and no strategy gets a feature the
others do not have.

The action is (ask logit, bid logit) in [-1, 1]^2. Depths are d_max * sigmoid(4 * logit), see
benchmark.config.depths_from_action, so the reachable depths are 0.018 * d_max to 0.982 * d_max
and the gradient stays informative everywhere.

Rewards and the transition law are identical to benchmark/simulator.py except for the optional
variance reduction described next; a test checks that the mean episode return matches the
Monte Carlo mean of the vectorised simulator for the same policy.

Variance reduction, and why it does not change the answer
--------------------------------------------------------
The realised marking term of a step is (q_{k+1} S_{k+1} - q_k S_k). Substituting the
conditional mean S_k for S_{k+1} removes (q_{k+1} (S_{k+1} - S_k)), a mean-zero martingale
increment which is independent of the fill indicators. Its expectation is zero under every
policy, so it does not move the expected objective and cannot move the optimal policy. It does
dominate the per-step signal to noise ratio (the inventory times the mid price move against
spread earnings of a similar size), which is why the default is noise_free=True. With
noise_free=True the expected step reward is p_ask d_ask + p_bid d_bid - phi q^2 dt, the same
expression the dynamic program uses, and the only remaining randomness is the fill indicator,
which the optimal policy genuinely has to trade off against the spread. Set noise_free=False to
train on the fully realised PnL; benchmark/run_all.py runs that ablation in both settings and
the README reports what it costs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from benchmark.config import Setting, depths_from_action


class MMBenchEnv(gym.Env):
    """Dealer episode for one trading day."""

    metadata = {'render_modes': []}

    def __init__(self, setting: Setting, seed: int = 0, noise_free: bool = True,
                 reward_scale: float = 1.0):
        super().__init__()
        self.setting = setting
        self.noise_free = noise_free
        # scaling the reward by a positive constant leaves every optimal policy unchanged; the
        # trainer divides by the known optimum so the critic regresses returns of order one,
        # which keeps the value loss from swamping the shared features early in training
        self.reward_scale = float(reward_scale)
        self.fill = setting.fill_model()
        self.observation_space = spaces.Box(low=np.array([0.0, -1.0], dtype=np.float32),
                                            high=np.array([1.0, 1.0], dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().reset(seed=seed)
        self.t_step = 0
        self.q = 0
        self.mid = setting.s0
        self.cash = 0.0

    def depths(self, action: np.ndarray) -> tuple[float, float]:
        """Map the action to (d_ask, d_bid) in price units."""
        d_ask, d_bid = depths_from_action(self.setting, action)
        return float(d_ask), float(d_bid)

    def fill_probabilities(self, d_ask: float, d_bid: float) -> tuple[float, float]:
        """Per-step fill probabilities, zero on a side that would breach the inventory bound."""
        dt = self.setting.dt
        p_ask = 1.0 - np.exp(-self.setting.lam * self.fill.f(d_ask) * dt)
        p_bid = 1.0 - np.exp(-self.setting.lam * self.fill.f(d_bid) * dt)
        if self.q <= -self.setting.Q:
            p_ask = 0.0
        if self.q >= self.setting.Q:
            p_bid = 0.0
        return float(p_ask), float(p_bid)

    def _observation(self) -> np.ndarray:
        return np.array([self.t_step * self.setting.dt / self.setting.T,
                         self.q / self.setting.Q], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t_step = 0
        self.q = 0
        self.mid = self.setting.s0
        self.cash = 0.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        setting = self.setting
        d_ask, d_bid = self.depths(action)
        p_ask, p_bid = self.fill_probabilities(d_ask, d_bid)
        filled_ask = bool(self.np_random.random() < p_ask)
        filled_bid = bool(self.np_random.random() < p_bid)

        cash_delta = 0.0
        if filled_ask:
            cash_delta += self.mid + d_ask
        if filled_bid:
            cash_delta -= self.mid - d_bid
        self.cash += cash_delta
        q_next = self.q + int(filled_bid) - int(filled_ask)

        mid_next = self.mid + setting.sigma * np.sqrt(setting.dt) * self.np_random.standard_normal()
        mark_price = self.mid if self.noise_free else mid_next
        reward = (cash_delta
                  + (q_next * mark_price - self.q * self.mid)
                  - setting.phi * self.q ** 2 * setting.dt)

        self.q = q_next
        self.mid = mid_next
        self.t_step += 1
        terminated = self.t_step >= setting.n_steps
        if terminated:
            reward -= setting.a * self.q ** 2
        info = {'fills': int(filled_ask) + int(filled_bid), 'inventory': self.q,
                'depth_ask': d_ask, 'depth_bid': d_bid}
        return self._observation(), float(reward * self.reward_scale), terminated, False, info
