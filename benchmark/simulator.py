"""Vectorised simulator for both settings.

Model per episode (one trading day, N = n_steps steps of length dt = T / N):

    mid price      S_{k+1} = S_k + sigma * sqrt(dt) * z_k,  z_k standard normal
    quotes         bid at S_k - d_bid, ask at S_k + d_ask, depths chosen from (k/T, q_k)
    fills          the ask fills with probability 1 - exp(-lam f(d_ask) dt), the bid
                   independently with 1 - exp(-lam f(d_bid) dt)
    cash           a sell fill adds S_k + d_ask, a buy fill subtracts S_k - d_bid
    penalty        -phi q_k^2 dt is charged each step on the inventory held at the start of
                   the step, matching the running reward of the HJB
    step reward    r_k = (cash change) + (q_{k+1} S_{k+1} - q_k S_k) - phi q_k^2 dt
    objective      J  = sum_k r_k - a q_N^2

Notes that matter for the benchmark
-----------------------------------
* The step reward telescopes: J = X_N + q_N S_N - q_0 S_0 - phi sum q_k^2 dt - a q_N^2.
  A test checks this identity directly, which validates the cash and marking accounting.
* The inventory boundaries are hard: at q = +Q the buy side is switched off, at q = -Q the
  sell side is switched off, so |q| never exceeds Q.
* Fills are priced at the maker's own quote, not at the touch: the whole spread crossed by
  the counterparty is earned. That is the standard market-making fill model and it is what
  makes the depth trade-off non-trivial.
* The mid price is a martingale with mean increments of zero and is independent of the fill
  process. Consequently, for any policy that does not condition on the price path, the
  expected objective does not depend on the price path at all: the shocks add dispersion to
  the realised PnL but move no probability mass in expectation. benchmark/dp.py therefore
  computes the exact optimal expected objective without simulating prices. The shocks are
  kept because they give the RL agent a noisy learning signal, as in the real problem, and
  because realised PnL dispersion is one of the reported metrics.
* The fill probability is the exact probability of at least one Poisson event in dt. It
  cannot produce two fills on the same side within one step, which understates fill counts by
  O((lam f dt)^2) relative to a true event-driven simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from benchmark.config import Setting


class Policy(Protocol):
    """Vectorised depth policy.

    Called with integer arrays of shape (n_ep,): the step index and the inventory. Must
    return (d_ask, d_bid) arrays of the same shape, in price units, already clipped by the
    caller.
    """

    def __call__(self, t_step: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class CRN:
    """Common random numbers shared by every strategy evaluated on the same episodes."""

    price_shocks: np.ndarray   # (n_steps, n_ep) standard normal
    fill_ask: np.ndarray       # (n_steps, n_ep) uniform on [0, 1)
    fill_bid: np.ndarray
    seed: int

    @property
    def n_episodes(self) -> int:
        return self.price_shocks.shape[1]


def make_crn(setting: Setting, n_episodes: int, seed: int) -> CRN:
    rng = np.random.default_rng(seed)
    shape = (setting.n_steps, n_episodes)
    return CRN(price_shocks=rng.standard_normal(shape),
               fill_ask=rng.random(shape),
               fill_bid=rng.random(shape),
               seed=seed)


@dataclass
class Rollout:
    pnl: np.ndarray            # (n_ep,) objective values J
    cash: np.ndarray           # (n_ep,) terminal cash X_N
    terminal_q: np.ndarray     # (n_ep,)
    mean_abs_q: np.ndarray     # (n_ep,) time average of |q|
    ask_fills: np.ndarray      # (n_ep,) number of sell fills
    bid_fills: np.ndarray
    steps_at_bound: np.ndarray
    terminal_s: np.ndarray     # (n_ep,) mid price at the terminal date
    penalty: np.ndarray        # (n_ep,) sum of phi q_k^2 dt
    mean_d_ask: float
    mean_d_bid: float

    @property
    def n_episodes(self) -> int:
        return self.pnl.size

    def summary(self) -> dict:
        pnl = self.pnl
        half = 1.959963984540054 * pnl.std(ddof=1) / np.sqrt(pnl.size)
        return dict(n_episodes=int(pnl.size),
                    mean_pnl=float(pnl.mean()),
                    std_pnl=float(pnl.std(ddof=1)),
                    ci95_half_width=float(half),
                    mean_abs_terminal_q=float(np.abs(self.terminal_q).mean()),
                    mean_abs_q=float(self.mean_abs_q.mean()),
                    asks_filled_per_episode=float(self.ask_fills.mean()),
                    bids_filled_per_episode=float(self.bid_fills.mean()),
                    mean_depth_ask=float(self.mean_d_ask),
                    mean_depth_bid=float(self.mean_d_bid),
                    steps_at_bound_fraction=float(
                        (self.steps_at_bound / pnl.size).mean()),
                    mean_penalty=float(self.penalty.mean()))


def simulate(setting: Setting, policy: Policy, crn: CRN, *,
             n_episodes: int | None = None, episode_slice: slice | None = None) -> Rollout:
    """Roll the policy out over the episodes of `crn` (or a subset of them)."""
    if episode_slice is None:
        episode_slice = slice(0, crn.n_episodes if n_episodes is None else n_episodes)
    shocks = crn.price_shocks[:, episode_slice]
    u_ask = crn.fill_ask[:, episode_slice]
    u_bid = crn.fill_bid[:, episode_slice]
    n_ep = shocks.shape[1]

    dt = setting.dt
    sqrt_dt = np.sqrt(dt)
    q_bound = setting.Q
    fill = setting.fill_model()

    q = np.zeros(n_ep, dtype=np.int64)
    s = np.full(n_ep, setting.s0)
    cash = np.zeros(n_ep)
    total = np.zeros(n_ep)
    abs_q_sum = np.zeros(n_ep)
    ask_fills = np.zeros(n_ep, dtype=np.int64)
    bid_fills = np.zeros(n_ep, dtype=np.int64)
    at_bound = np.zeros(n_ep, dtype=np.int64)
    d_ask_sum = 0.0
    d_bid_sum = 0.0
    penalty = np.zeros(n_ep)

    for step in range(setting.n_steps):
        d_ask, d_bid = policy(np.full(n_ep, step, dtype=np.int64), q)
        d_ask = np.clip(np.asarray(d_ask, dtype=float), 0.0, setting.d_max)
        d_bid = np.clip(np.asarray(d_bid, dtype=float), 0.0, setting.d_max)
        d_ask_sum += float(d_ask.mean())
        d_bid_sum += float(d_bid.mean())

        f_ask = fill.f(d_ask)
        f_bid = fill.f(d_bid)
        p_ask = 1.0 - np.exp(-setting.lam * f_ask * dt)
        p_buy = 1.0 - np.exp(-setting.lam * f_bid * dt)
        p_ask = np.where(q <= -q_bound, 0.0, p_ask)
        p_buy = np.where(q >= q_bound, 0.0, p_buy)

        filled_ask = u_ask[step] < p_ask
        filled_bid = u_bid[step] < p_buy
        ask_fills += filled_ask
        bid_fills += filled_bid

        cash_delta = filled_ask * (s + d_ask) - filled_bid * (s - d_bid)
        cash = cash + cash_delta
        q_next = q + filled_bid.astype(np.int64) - filled_ask.astype(np.int64)
        s_next = s + setting.sigma * sqrt_dt * shocks[step]

        step_penalty = setting.phi * q.astype(float) ** 2 * dt
        penalty += step_penalty
        total += cash_delta + (q_next * s_next - q * s) - step_penalty
        abs_q_sum += np.abs(q)
        at_bound += np.abs(q) >= q_bound
        q = q_next
        s = s_next

    total -= setting.a * q.astype(float) ** 2
    return Rollout(pnl=total, cash=cash, terminal_q=q, mean_abs_q=abs_q_sum / setting.n_steps,
                   ask_fills=ask_fills, bid_fills=bid_fills, steps_at_bound=at_bound,
                   terminal_s=s, penalty=penalty,
                   mean_d_ask=d_ask_sum / setting.n_steps,
                   mean_d_bid=d_bid_sum / setting.n_steps)
