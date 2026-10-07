"""Optimal-policy solver for the two setting families.

Model
-----
State: time t in [0, T] and integer inventory q in {-Q, ..., Q}. The unit of inventory is one
share in the equity setting and one million of face value in the credit setting.
Controls: bid depth d_b >= 0 and ask depth d_a >= 0, both measured from the mid price S.
Fills: each side fills as a Poisson process with intensity lam * f(d).
Objective (risk neutral, quadratic running penalty, terminal inventory penalty):

    J = E[ X_T + q_T S_T - phi * integral_0^T q_t^2 dt - a * q_T^2 ]

Derivation (re-done from scratch, then checked against the closed form in tests)
-----------------------------------------------------------------------------
Write the value function as V(t, x, s, q) = x + q s + h(t, q). The mid price is an
arithmetic Brownian motion, so the drift and diffusion of q s contribute nothing to the
generator over dt (q s is linear in s). A bid fill (we buy one lot at s - d_b) moves
(x, q) -> (x - (s - d_b), q + 1) and its contribution to the generator is

    lam f(d_b) [ V(t, x - (s - d_b), s, q + 1) - V(t, x, s, q) ]
  = lam f(d_b) [ d_b - ( h(t, q) - h(t, q + 1) ) ]

An ask fill (we sell one lot at s + d_a) moves (x, q) -> (x + (s + d_a), q - 1) and gives

    lam f(d_a) [ d_a - ( h(t, q) - h(t, q - 1) ) ]

The HJB is 0 = -d_t V + generator + running reward, hence with H(p) = sup_{d >= 0} f(d)(d - p):

    d_t h(t, q) = phi q^2 - lam H( h(t,q) - h(t,q-1) ) - lam H( h(t,q) - h(t,q+1) )
    h(T, q) = -a q^2

which is the standard statement of this problem. The optimal depths are the maximisers
inside H:

    ask: p_ask = h(t,q) - h(t,q-1)      bid: p_bid = h(t,q) - h(t,q+1)
    d_ask = argmax_{d>=0} f(d)(d - p_ask)   d_bid = argmax_{d>=0} f(d)(d - p_bid)

At the inventory bound the transition that would breach it is removed: at q = +Q the bid
term is dropped, at q = -Q the ask term is dropped. That is the same truncation the closed
form applies by exponentiating a finite tridiagonal matrix.

Finding: the depth floor matters, so this module does not use the textbook closed form as
the answer. H(d) is maximised over d >= 0, and for the parameters used here an inventory of
+-Q makes the risk price of the transition that reduces |q| more negative than -1/k. The
exact optimum then quotes at the mid on that side, where the exponential branch of H is
outside its valid range. The Cartea-Jaimungal closed form solves the corresponding
unfloored linear system, so it agrees with the exact solution near q = 0 (to hundredths of
a price unit) and differs near the bound by tens of units. tests/test_hjb.py pins both
facts. The optimum reported by this module is the floored one.

Envelope theorem: H'(p) = -f(d*(p)), which gives an exact analytic Jacobian for the solver.

Worked example (exponential fills, k = 1.5, p = 0.2):
    d* = argmax over d >= 0 of exp(-k d)(d - p) satisfies 1 - k(d - p) = 0, so d* = p + 1/k
       = 0.2 + 0.667 = 0.867
    H(0.2) = exp(-1.5 * 0.867) * 0.667 = 0.2725 * 0.667 = 0.1818
    and the closed form (1/(k e)) exp(-k p) = (1/(1.5 * 2.71828)) * exp(-0.3) = 0.2453 * 0.7408
       = 0.1818, the same number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit


class ExponentialFill:
    """f(d) = exp(-k d). Closed-form H so the solver can be validated exactly."""

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError('k must be positive')
        self.k = float(k)

    def f(self, d):
        return np.exp(-self.k * np.asarray(d, dtype=float))

    def optimal_depth(self, p):
        """argmax_{d>=0} exp(-k d) (d - p) = max(0, p + 1/k), floored at 0."""
        p = np.asarray(p, dtype=float)
        return np.maximum(0.0, p + 1.0 / self.k)

    def H(self, p):
        """sup_{d>=0} f(d)(d-p) = (1/(k e)) exp(-k p) while p >= -1/k, else -p (d = 0)."""
        p = np.asarray(p, dtype=float)
        interior = (1.0 / (self.k * np.e)) * np.exp(-self.k * p)
        boundary = -p
        return np.where(p + 1.0 / self.k > 0.0, interior, boundary)

    def dH_dp(self, p):
        return -self.f(self.optimal_depth(p))

    def optimal_depth_is_interior(self, p):
        return np.asarray(p, dtype=float) + 1.0 / self.k > 0.0


class LogisticFill:
    """f(d) = 1 / (1 + exp(alpha + beta d)), the corporate-bond RFQ hit ratio.

    beta > 0 so the hit ratio decays with the quoted half-spread. The maximiser solves
    beta (1 - f(d)) (d - p) = 1 by the first-order condition; the solver brackets and
    bisects, and tests/test_hjb.py checks it against a brute-force grid search.
    """

    def __init__(self, alpha: float, beta: float):
        if beta <= 0:
            raise ValueError('beta must be positive so deeper quotes are less likely to hit')
        self.alpha = float(alpha)
        self.beta = float(beta)

    def f(self, d):
        d = np.asarray(d, dtype=float)
        return expit(-(self.alpha + self.beta * d))

    def _foc(self, d, p):
        return self.beta * (1.0 - self.f(d)) * (d - p) - 1.0

    def _unconstrained_root(self, p, n_iter=100):
        p = np.asarray(p, dtype=float)
        span = (2.0 + abs(self.alpha) + 20.0) / self.beta + 2.0
        lo = np.minimum(p, 0.0) - span
        hi = np.maximum(p, 0.0) + span
        # grow the bracket until the FOC is positive at hi
        for _ in range(200):
            bad = self._foc(hi, p) <= 0.0
            if not np.any(bad):
                break
            hi = np.where(bad, hi * 2.0 + 1.0, hi)
        else:  # pragma: no cover - defensive, span already generous
            raise RuntimeError('failed to bracket the logistic first-order condition')
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            positive = self._foc(mid, p) > 0.0
            hi = np.where(positive, mid, hi)
            lo = np.where(positive, lo, mid)
        return 0.5 * (lo + hi)

    def optimal_depth(self, p):
        """argmax_{d>=0} f(d)(d-p), floored at 0 like the exponential case."""
        return np.maximum(0.0, self._unconstrained_root(p))

    def H(self, p):
        p = np.asarray(p, dtype=float)
        d = self.optimal_depth(p)
        return self.f(d) * (d - p)

    def dH_dp(self, p):
        return -self.f(self.optimal_depth(p))

    def optimal_depth_is_interior(self, p):
        return self._unconstrained_root(p) > 0.0


def make_fill(fill_kind: str, k: float | None = None, alpha: float | None = None,
              beta: float | None = None):
    if fill_kind == 'exponential':
        return ExponentialFill(k=k)
    if fill_kind == 'logistic':
        return LogisticFill(alpha=alpha, beta=beta)
    raise ValueError(f'unknown fill kind {fill_kind!r}')


def h_terms(h, fill):
    """The two H terms of the ODE, one array per side, at every inventory index.

    Index i is inventory q_i = -Q + i. The ask fill (we sell, q_i -> q_i - 1) is only
    available for i >= 1 and has price of inventory risk p_ask(i) = h[i] - h[i-1]. The bid
    fill (we buy, q_i -> q_i + 1) is only available for i <= n - 2 and has
    p_bid(i) = h[i] - h[i+1]. Disabled sides contribute exactly 0, which is the truncation
    that keeps the inventory inside the grid.
    """
    h = np.asarray(h, dtype=float)
    n = h.shape[0]
    h_ask = np.zeros(n)
    h_bid = np.zeros(n)
    h_ask[1:] = fill.H(h[1:] - h[:-1])
    h_bid[:-1] = fill.H(h[:-1] - h[1:])
    return h_ask, h_bid


def depths_at(h, fill):
    """Optimal depths implied by a value function array of shape (..., n_q).

    Returns (d_ask, d_bid) with nan wherever the side is disabled, using the same index
    convention as h_terms.
    """
    h = np.asarray(h, dtype=float)
    d_ask = np.full(h.shape, np.nan)
    d_bid = np.full(h.shape, np.nan)
    d_ask[..., 1:] = fill.optimal_depth(h[..., 1:] - h[..., :-1])
    d_bid[..., :-1] = fill.optimal_depth(h[..., :-1] - h[..., 1:])
    return d_ask, d_bid


def hjb_rhs(h, fill, phi, lam, q_grid):
    """Right-hand side of the backward ODE, vectorised over q."""
    h_ask, h_bid = h_terms(h, fill)
    return phi * np.asarray(q_grid, dtype=float) ** 2 - lam * (h_ask + h_bid)


def hjb_jacobian(h, fill, phi, lam, q_grid):
    """Analytic Jacobian d(rhs)/dh using the envelope theorem H'(p) = -f(d*(p)).

    Row i holds -(lam f_ask[i]) at column i - 1, +(lam (f_ask[i] + f_bid[i])) at column i and
    -(lam f_bid[i]) at column i + 1, with the disabled sides left at zero.
    """
    h = np.asarray(h, dtype=float)
    n = h.shape[0]
    f_ask = np.zeros(n)
    f_bid = np.zeros(n)
    f_ask[1:] = fill.f(fill.optimal_depth(h[1:] - h[:-1]))
    f_bid[:-1] = fill.f(fill.optimal_depth(h[:-1] - h[1:]))
    diag = lam * (f_ask + f_bid)
    jac = np.diag(diag)
    upper = np.arange(1, n)
    jac[upper, upper - 1] -= lam * f_ask[upper]
    jac[upper - 1, upper] -= lam * f_bid[upper - 1]
    return jac


@dataclass
class HJBResult:
    t: np.ndarray          # shape (n_t,), ascending from 0 to T
    q: np.ndarray          # shape (2Q+1,)
    h: np.ndarray          # shape (n_t, 2Q+1)
    d_ask: np.ndarray      # shape (n_t, 2Q+1), nan where the side is disabled
    d_bid: np.ndarray
    fill: object
    phi: float
    a: float
    lam: float

    def value(self, t, q_index):
        """h(t, q) at the stored grid, by nearest stored time index."""
        i = int(np.argmin(np.abs(self.t - t)))
        return self.h[i, q_index]


def solve_hjb(fill, T, Q, phi, a, lam, n_t=401, rtol=1e-10, atol=1e-12,
              method='Radau', t_eval=None):
    """Solve the backward ODE on a uniform time grid.

    Integrates from t = T (terminal condition h = -a q^2) back to t = 0 with an implicit
    scheme and the analytic Jacobian. Returns h plus the optimal depths on the grid.
    Pass t_eval to report the solution on a caller supplied ascending time grid.
    """
    q_grid = np.arange(-Q, Q + 1, dtype=float)
    h_T = -a * q_grid ** 2
    if t_eval is None:
        t_eval = np.linspace(0.0, T, n_t)
    t_eval = np.asarray(t_eval, dtype=float)
    solution = solve_ivp(
        fun=lambda t, y: hjb_rhs(y, fill, phi, lam, q_grid),
        t_span=(T, 0.0),
        y0=h_T,
        method=method,
        jac=lambda t, y: hjb_jacobian(y, fill, phi, lam, q_grid),
        t_eval=t_eval[::-1],
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f'HJB integration failed: {solution.message}')
    # solve_ivp returns points in the order of t_eval, which was T -> 0; flip to ascending
    t_ascending = solution.t[::-1].copy()
    h_ascending = solution.y.T[::-1].copy()

    d_ask, d_bid = depths_at(h_ascending, fill)
    return HJBResult(t=t_ascending, q=q_grid, h=h_ascending, d_ask=d_ask, d_bid=d_bid,
                     fill=fill, phi=phi, a=a, lam=lam)


def closed_form_exponential(T, Q, phi, a, lam, k, t_grid):
    """Cartea-Jaimungal solution for exponential fills.

    h(t, q) = (1/k) log w_q(t),  w(t) = expm(M (T - t)) z,
    M[q,q] = -phi k q^2, M[q,q+-1] = lam / e,  z_q = exp(-a k q^2).
    """
    from scipy.linalg import expm

    q_grid = np.arange(-Q, Q + 1)
    n = q_grid.size
    M = np.zeros((n, n))
    for i, q in enumerate(q_grid):
        M[i, i] = -phi * k * q ** 2
        if i + 1 < n:
            M[i, i + 1] = lam / np.e
        if i - 1 >= 0:
            M[i, i - 1] = lam / np.e
    z = np.exp(-a * k * q_grid ** 2)
    t_grid = np.asarray(t_grid, dtype=float)
    w = np.stack([expm(M * (T - t)) @ z for t in t_grid])
    h = np.log(w) / k
    # Written out independently of depths_at so the test compares two separate code paths.
    d_ask = np.full_like(h, np.nan)
    d_bid = np.full_like(h, np.nan)
    d_ask[:, 1:] = np.maximum(0.0, 1.0 / k + h[:, 1:] - h[:, :-1])
    d_bid[:, :-1] = np.maximum(0.0, 1.0 / k + h[:, :-1] - h[:, 1:])
    return h, d_ask, d_bid


def as_heuristic_depths(t, q, S, gamma, sigma, k, T):
    """Avellaneda-Stoikov (2008) quotes expressed as depths from the mid price.

    Reservation price r = S - q gamma sigma^2 (T - t).
    Total spread = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k).
    Quoting symmetrically around r gives
        d_ask = (r - S) + spread / 2 = -q gamma sigma^2 (T - t) + spread / 2
        d_bid = (S - r) + spread / 2 = +q gamma sigma^2 (T - t) + spread / 2
    Negative depths are floored at 0, which means quoting at the mid on that side.

    Note: this is an asymptotic approximation to an exponential-utility problem, so it is a
    strong heuristic for this benchmark but not the optimum of the quadratic-penalty
    objective solved by solve_hjb.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    tau = T - t
    skew = q * gamma * sigma ** 2 * tau
    spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
    d_ask = np.maximum(0.0, -skew + 0.5 * spread)
    d_bid = np.maximum(0.0, skew + 0.5 * spread)
    return d_ask, d_bid
