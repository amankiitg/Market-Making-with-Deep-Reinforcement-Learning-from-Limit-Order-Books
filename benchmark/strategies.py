"""Depth policies, all sharing one interface: (t_step, q) arrays in, (d_ask, d_bid) out.

A depth that is forced to a switched off side never enters the objective, so table policies
substitute d_max for the nan entries that mark a switched off side. The simulator clips to
[0, d_max] and sets the fill probability at a switched off side to zero regardless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.config import Setting, depths_from_action
from benchmark.hjb import as_heuristic_depths, closed_form_exponential, solve_hjb


def step_time(setting: Setting) -> np.ndarray:
    """Time at the start of each simulator step, plus the terminal time (n_steps + 1 rows)."""
    return np.linspace(0.0, setting.T, setting.n_steps + 1)


class TablePolicy:
    """Look up depths from an array with one row per step and one column per inventory."""

    def __init__(self, d_ask: np.ndarray, d_bid: np.ndarray, setting: Setting, name: str):
        self.setting = setting
        self.name = name
        self.d_ask = np.asarray(d_ask, dtype=float)
        self.d_bid = np.asarray(d_bid, dtype=float)
        if self.d_ask.shape[0] < setting.n_steps:
            raise ValueError(f'{name}: table needs at least {setting.n_steps} rows')

    def depths(self, step: int, q: int) -> tuple[float, float]:
        col = int(np.clip(q + self.setting.Q, 0, 2 * self.setting.Q))
        a = self.d_ask[step, col]
        b = self.d_bid[step, col]
        fallback = self.setting.d_max
        return (float(a) if np.isfinite(a) else fallback,
                float(b) if np.isfinite(b) else fallback)

    def __call__(self, t_step: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.clip(q + self.setting.Q, 0, 2 * self.setting.Q)
        a = self.d_ask[t_step, col]
        b = self.d_bid[t_step, col]
        a = np.where(np.isfinite(a), a, self.setting.d_max)
        b = np.where(np.isfinite(b), b, self.setting.d_max)
        return a, b


@dataclass
class ConstantPolicy:
    """Quote the same depth on both sides for the whole episode."""

    setting: Setting
    depth: float
    name: str = 'constant'

    def __call__(self, t_step: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = np.shape(t_step)
        return np.full(shape, self.depth), np.full(shape, self.depth)


class ASHeuristicPolicy:
    """Avellaneda-Stoikov reservation price heuristic, floored at a zero depth.

    Only defined for the exponential setting because the total-spread term contains 1 / k.
    """

    name = 'avellaneda_stoikov_heuristic'

    def __init__(self, setting: Setting):
        if setting.fill_kind != 'exponential':
            raise ValueError('the Avellaneda-Stoikov heuristic needs an exponential fill model')
        self.setting = setting

    def __call__(self, t_step: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = t_step * self.setting.dt
        return as_heuristic_depths(t, q, self.setting.s0, self.setting.gamma,
                                   self.setting.sigma, self.setting.k, self.setting.T)


def hjb_table(setting: Setting, n_t: int | None = None):
    """Solve the HJB and return (result, ask table, bid table) on the simulator time grid."""
    fill = setting.fill_model()
    t_grid = step_time(setting) if n_t is None else np.linspace(0.0, setting.T, n_t)
    result = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                       lam=setting.lam, t_eval=t_grid)
    return result, result.d_ask, result.d_bid


def hjb_policy(setting: Setting) -> TablePolicy:
    _, d_ask, d_bid = hjb_table(setting)
    return TablePolicy(d_ask, d_bid, setting, name='hjb_optimal')


def closed_form_policy(setting: Setting) -> TablePolicy:
    """Textbook Cartea-Jaimungal closed form, floored at a zero depth.

    Available for the exponential setting only. It solves the unfloored problem, so it is a
    strong but strictly inferior reference next to the HJB table.
    """
    if setting.fill_kind != 'exponential':
        raise ValueError('the closed form needs an exponential fill model')
    t_grid = step_time(setting)
    h, d_ask, d_bid = closed_form_exponential(T=setting.T, Q=setting.Q, phi=setting.phi,
                                             a=setting.a, lam=setting.lam, k=setting.k,
                                             t_grid=t_grid)
    return TablePolicy(d_ask, d_bid, setting, name='closed_form_cartea_jaimungal')


def dp_policy(dp_result, setting: Setting) -> TablePolicy:
    return TablePolicy(dp_result.d_ask, dp_result.d_bid, setting, name='dp_optimal')


class RLPolicy:
    """Wrap a trained model that maps (t / T, q / Q) to a logit action pair.

    Action order is (ask, bid), matching the (d_ask, d_bid) convention of every policy here
    and of benchmark/simulator.py, and the squash from action to depth is
    benchmark.config.depths_from_action, the same function the environment uses.
    """

    name = 'rl'

    def __init__(self, model, setting: Setting, deterministic: bool = True):
        self.model = model
        self.setting = setting
        self.deterministic = deterministic

    def observations(self, t_step: np.ndarray, q: np.ndarray) -> np.ndarray:
        t = np.asarray(t_step, dtype=float) * self.setting.dt / self.setting.T
        inv = np.asarray(q, dtype=float) / self.setting.Q
        return np.stack([t, inv], axis=1)

    def __call__(self, t_step: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actions, _ = self.model.predict(self.observations(t_step, q),
                                       deterministic=self.deterministic)
        return depths_from_action(self.setting, actions)
