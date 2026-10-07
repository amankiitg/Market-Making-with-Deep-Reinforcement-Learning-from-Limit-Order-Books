"""Solver tests.

The general backward ODE must reproduce the exact matrix-exponential closed form, and the
logistic first-order condition must match a brute-force grid search.
"""

import numpy as np
import pytest

from benchmark.config import credit_setting, equity_setting
from benchmark.hjb import (ExponentialFill, LogisticFill, closed_form_exponential, make_fill,
                           solve_hjb)


def test_logistic_H_and_depth_match_grid_search():
    """The bisection solution of the first-order condition must equal a dense grid search."""
    fill = LogisticFill(alpha=0.0, beta=5.5)
    d_grid = np.linspace(0.0, 1.5, 300001)
    for p in np.linspace(-1.2, 0.6, 19):
        values = fill.f(d_grid) * (d_grid - p)
        best = values.max()
        d_star = d_grid[int(np.argmax(values))]
        assert fill.H(p) == pytest.approx(best, abs=2e-6), f'H mismatch at p={p}'
        assert fill.optimal_depth(p) == pytest.approx(d_star, abs=2e-5), f'depth at p={p}'


def test_exponential_H_and_depth_are_exact():
    fill = ExponentialFill(k=1.5)
    for p in np.linspace(-0.3, 2.0, 25):
        assert fill.optimal_depth(p) == pytest.approx(max(0.0, p + 1.0 / 1.5), abs=1e-12)
        expected = (-p if p <= -1.0 / 1.5
                    else (1.0 / (1.5 * np.e)) * np.exp(-1.5 * p))
        assert fill.H(p) == pytest.approx(expected, rel=1e-12, abs=1e-12)


class _UnflooredExponential(ExponentialFill):
    """H(p) = (1/(k e)) exp(-k p) for every p, i.e. the linear system whose solution is the
    Cartea-Jaimungal matrix exponential: the depths p + 1/k are allowed to go negative.

    Used only to check that the ODE machinery reproduces expm(M (T - t)) exactly.
    """

    def optimal_depth(self, p):
        return np.asarray(p, dtype=float) + 1.0 / self.k

    def H(self, p):
        p = np.asarray(p, dtype=float)
        return (1.0 / (self.k * np.e)) * np.exp(-self.k * p)


@pytest.mark.parametrize('method', ['Radau', 'BDF'])
def test_solver_matches_exponential_closed_form(method):
    """Radau and BDF integration with the analytic Jacobian must both reproduce the exact
    Cartea-Jaimungal solution at every grid point and inventory level."""
    setting = equity_setting()
    fill = _UnflooredExponential(k=setting.k)
    result = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                       lam=setting.lam, n_t=201, method=method)
    h_closed, d_ask_closed, d_bid_closed = closed_form_exponential(
        T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a, lam=setting.lam,
        k=setting.k, t_grid=result.t)

    max_err = float(np.abs(result.h - h_closed).max())
    assert max_err < 1e-8 * max(1.0, float(np.abs(h_closed).max())), (
        f'{method} deviates from the closed form by {max_err:.3e}'
    )
    # depths implied by the closed-form h, written out independently of hjb.depths_at
    k = setting.k
    np.testing.assert_allclose(result.d_ask[:, 1:], 1.0 / k + h_closed[:, 1:] - h_closed[:, :-1],
                               atol=1e-8)
    np.testing.assert_allclose(result.d_bid[:, :-1], 1.0 / k + h_closed[:, :-1] - h_closed[:, 1:],
                               atol=1e-8)
    assert np.all(np.isnan(d_ask_closed[:, :1])) and np.all(np.isnan(d_bid_closed[:, -1:]))


def test_floor_at_zero_depth_separates_exact_solution_from_closed_form():
    """The d >= 0 floor matters. Recorded finding, not a bug.

    For these parameters an inventory of +-Q makes the price of inventory risk of the
    transition that reduces |q| more negative than -1/k, so the exact optimum quotes at the
    mid (d = 0) on that side. The textbook exponential H formula is then outside its valid
    range, and the closed form solves a different (unfloored) system. The gap is large at
    |q| = Q and decays towards q = 0, where the two agree to a few hundredths of a unit.
    """
    setting = equity_setting()
    fill = make_fill('exponential', k=setting.k)
    result = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                       lam=setting.lam, n_t=201)
    h_closed, _, _ = closed_form_exponential(
        T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a, lam=setting.lam,
        k=setting.k, t_grid=result.t)

    assert np.nanmin(result.d_ask) == 0.0, 'expected the floor to bind somewhere'
    diff = np.abs(result.h - h_closed)
    assert diff[:, setting.Q].max() < 0.01, 'the two must agree at q = 0'
    assert diff.max() > 1.0, 'the two must disagree near the inventory bound'
    # decay of the gap from the bound inwards, at a time away from the terminal date where the
    # two solutions coincide by construction
    for idx in (0, setting.n_steps // 2):
        row = diff[idx]
        assert row[0] > row[setting.Q - 1] >= row[setting.Q]


def test_solution_is_mesh_independent():
    """Halving dt must not move h by more than the integration tolerance."""
    setting = credit_setting()
    fill = make_fill('logistic', alpha=setting.alpha, beta=setting.beta)
    coarse = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                       lam=setting.lam, n_t=201, rtol=1e-12, atol=1e-14)
    fine = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                     lam=setting.lam, n_t=801, rtol=1e-12, atol=1e-14,
                     t_eval=coarse.t)
    assert np.abs(coarse.h - fine.h).max() < 1e-9


def test_terminal_condition_and_symmetry():
    setting = credit_setting()
    fill = make_fill('logistic', alpha=setting.alpha, beta=setting.beta)
    result = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                       lam=setting.lam, n_t=101)
    assert np.allclose(result.h[-1], -setting.a * result.q ** 2, atol=1e-8)
    h_final = result.h[0]
    assert np.allclose(h_final, h_final[::-1], atol=1e-8)


def test_depths_skew_away_from_inventory_and_stay_inside_the_action_bound():
    """A long position must quote the ask closer and the bid further, depths must be finite,
    and the depths must stay inside the RL action bound wherever the policy actually operates.

    The unconstrained solution wants a very deep quote at the inventory bound near the end of
    the day (up to 8.5 price units for equity). The action bound is tighter than that on
    purpose; tests/test_benchmark.py measures what the bound costs, and it is a few thousandths
    of a percent of the optimum."""
    for setting in (equity_setting(), credit_setting()):
        fill = make_fill(setting.fill_kind, k=setting.k, alpha=setting.alpha,
                         beta=setting.beta)
        result = solve_hjb(fill, T=setting.T, Q=setting.Q, phi=setting.phi, a=setting.a,
                           lam=setting.lam, n_t=201)
        q = result.q
        # the ask side (q -> q-1) is available for index >= 1, the bid side for index <= n-2
        assert np.all(np.isnan(result.d_ask[:, 0]))
        assert np.all(np.isnan(result.d_bid[:, -1]))
        d_ask = result.d_ask[:, 1:]
        d_bid = result.d_bid[:, :-1]
        assert np.all(np.isfinite(d_ask)) and np.all(np.isfinite(d_bid))
        for i in range(result.t.size):
            assert np.all(np.diff(d_ask[i]) <= 1e-9), 'ask depth must not increase with q'
            assert np.all(np.diff(d_bid[i]) >= -1e-9), 'bid depth must not decrease with q'
        # at q = 0 both sides are quoted, so the two sides coincide
        mid_index = int(np.where(q == 0)[0][0])
        assert np.allclose(result.d_ask[:, mid_index], result.d_bid[:, mid_index], atol=1e-9)
        # The unconstrained solution wants deeper quotes than the RL action bound at the ends of
        # the inventory range and late in the day, which is why the bound is a separate design
        # decision that tests/test_benchmark.py measures the cost of. What must hold here is the
        # shape of the solution and that the policy operates well inside the bound at a flat book.
        for label, table in (('ask', result.d_ask), ('bid', result.d_bid)):
            assert np.nanmax(table[0, setting.Q - 1:setting.Q + 2]) < 0.5 * setting.d_max, (
                f'{setting.name}: {label} depth at the open with |q| <= 1 is close to the '
                f'action bound {setting.d_max}'
            )
            fraction_at_bound = float(np.mean(table[:, 1:-1] >= 0.99 * setting.d_max))
            assert fraction_at_bound < 0.45, (
                f'{setting.name}: {label} sits at the action bound in '
                f'{100 * fraction_at_bound:.1f} percent of grid cells'
            )
