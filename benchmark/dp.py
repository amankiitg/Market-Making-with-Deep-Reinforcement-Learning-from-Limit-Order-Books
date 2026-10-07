"""Exact backward dynamic program for the discrete simulator.

Why this is the grading anchor
-----------------------------
The simulator moves the inventory by at most one lot up and one lot down per step, with
side probabilities

    p_ask(d) = 1 - exp(-lam f(d) dt)     p_bid(d) = 1 - exp(-lam f(d) dt)

and the cash and marking terms of the step reward cancel in expectation when they are
written relative to the mid, because the mid is a martingale that is independent of the
fill process. So the state is just (step, inventory) and the expected objective satisfies a
finite backwards recursion. That recursion can be solved exactly on the inventory grid, and
its value at (0, 0) is the largest achievable expected J. It is the number the RL agent is
compared against, and it needs no Monte Carlo, so it carries no sampling noise.

Decomposition used to keep the optimisation cheap
------------------------------------------------
Writing V for the value at step k + 1 and dropping the constant and the running penalty,

    objective = p_a (d_a + V_{q-1} - V_q) + p_b (d_b + V_{q+1} - V_q)
                + p_a p_b (2 V_q - V_{q+1} - V_{q-1})

so for a fixed d_b the ask problem is exactly

    max over d_a of p_a(d_a) (d_a - P_a),  P_a = V_q - V_{q-1} - p_b B_q,
    B_q = 2 V_q - V_{q+1} - V_{q-1}

with the symmetric statement for the bid. The two sides therefore decouple given the other
side's probability, which a handful of sweeps resolves. B_q is the curvature of V in the
inventory, which is non-negative by concavity, and p_a p_b = O((lam dt)^2), so the coupling
is genuinely small. Each one dimensional problem is solved on a depth grid followed by a
parabolic refinement.

Boundaries: at q = -Q the ask side is switched off (p_a = 0) and at q = +Q the bid side is
switched off, exactly as in simulator.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.config import Setting


@dataclass
class DPResult:
    value: np.ndarray     # (n_steps + 1, 2Q+1), expected objective from (step, q)
    d_ask: np.ndarray     # (n_steps, 2Q+1), nan where the side is switched off
    d_bid: np.ndarray
    q: np.ndarray
    setting_name: str

    @property
    def optimal_value(self) -> float:
        """Exact optimal expected objective from an empty book at step 0."""
        return float(self.value[0, self.q.size // 2])


def _parabolic_refine(x0, x1, x2, y0, y1, y2):
    """Vertex of the parabola through three equally spaced points, clipped to the bracket."""
    denom = (y0 - 2.0 * y1 + y2)
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = np.where(np.abs(denom) > 1e-300, 0.5 * (y0 - y2) / denom, 0.0)
    return np.clip(x1 + offset * (x1 - x0), x0, x2)


def _maximise(p_of_d, d_grid, p_side):
    """max over depth of p(d) (d - p_side), vectorised over the inventory axis.

    p_of_d is (n_d,) and p_side is (n_q, 1). Returns (d_star, best_value) with arrays of
    length n_q. The grid optimum is refined with the vertex of the parabola through its two
    neighbours, and the refinement is kept only if it improves the objective.
    """
    values = (d_grid[None, :] - p_side) * p_of_d[None, :]
    idx = np.argmax(values, axis=1)
    rows = np.arange(values.shape[0])
    best = values[rows, idx]
    interior = (idx > 0) & (idx < d_grid.size - 1)
    if np.any(interior):
        i0 = np.maximum(idx - 1, 0)
        i2 = np.minimum(idx + 1, d_grid.size - 1)
        y0 = values[rows, i0]
        y2 = values[rows, i2]
        d_star = _parabolic_refine(d_grid[i0], d_grid[idx], d_grid[i2], y0, best, y2)
        refined = np.interp(d_star, d_grid, p_of_d) * (d_star - p_side[:, 0])
        better = interior & (refined > best)
        best = np.where(better, refined, best)
        d_star = np.where(better, d_star, d_grid[idx])
    else:  # pragma: no cover - needs a degenerate grid
        d_star = d_grid[idx]
    return d_star, best


def solve_dp(setting: Setting, n_depth: int = 4001, n_sweeps: int = 4) -> DPResult:
    """Solve the exact dynamic program of the discrete simulator."""
    dt = setting.dt
    q_grid = np.arange(-setting.Q, setting.Q + 1)
    n_q = q_grid.size
    q_float = q_grid.astype(float)
    fill = setting.fill_model()

    d_grid = np.linspace(0.0, setting.d_max, n_depth)
    p_of_d = 1.0 - np.exp(-setting.lam * fill.f(d_grid) * dt)   # (n_d,)

    value = np.empty((setting.n_steps + 1, n_q))
    value[-1] = -setting.a * q_float ** 2
    d_ask = np.full((setting.n_steps, n_q), np.nan)
    d_bid = np.full((setting.n_steps, n_q), np.nan)

    ask_available = q_grid > -setting.Q
    bid_available = q_grid < setting.Q

    for step in range(setting.n_steps - 1, -1, -1):
        v_next = value[step + 1]
        # zero weight padding so the index shifts stay in range; every weight that would
        # touch the padding is zero because a switched off side has probability zero
        v_pad = np.concatenate([v_next[:1], v_next, v_next[-1:]])
        v_up = v_pad[2:]      # V_{q+1}
        v_down = v_pad[:-2]   # V_{q-1}
        curvature = 2.0 * v_next - v_up - v_down

        p_a = np.zeros(n_q)
        p_b = np.zeros(n_q)
        d_a = np.zeros(n_q)
        d_b = np.zeros(n_q)
        for _ in range(n_sweeps):
            p_side_a = (v_next - v_down - p_b * curvature)[:, None]
            d_a, _ = _maximise(p_of_d, d_grid, p_side_a)
            p_a = np.where(ask_available, 1.0 - np.exp(-setting.lam * fill.f(d_a) * dt), 0.0)
            p_side_b = (v_next - v_up - p_a * curvature)[:, None]
            d_b, _ = _maximise(p_of_d, d_grid, p_side_b)
            p_b = np.where(bid_available, 1.0 - np.exp(-setting.lam * fill.f(d_b) * dt), 0.0)

        expectation = (v_next
                       + p_b * (v_up - v_next)
                       + p_a * (v_down - v_next)
                       + p_a * p_b * curvature)
        value[step] = expectation + p_a * d_a + p_b * d_b - setting.phi * q_float ** 2 * dt
        d_a = np.where(ask_available, d_a, np.nan)
        d_b = np.where(bid_available, d_b, np.nan)
        d_ask[step] = d_a
        d_bid[step] = d_b

    return DPResult(value=value, d_ask=d_ask, d_bid=d_bid, q=q_grid,
                    setting_name=setting.name)
