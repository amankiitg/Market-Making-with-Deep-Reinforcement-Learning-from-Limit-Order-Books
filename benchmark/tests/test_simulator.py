"""Simulator accounting and the dynamic-programming ground truth."""

from dataclasses import replace

import numpy as np
import pytest

from benchmark.config import credit_setting, equity_setting
from benchmark.dp import solve_dp
from benchmark.simulator import make_crn, simulate
from benchmark.strategies import ASHeuristicPolicy, ConstantPolicy, dp_policy, hjb_policy


def _se(rollout):
    return rollout.pnl.std(ddof=1) / np.sqrt(rollout.n_episodes)


def test_step_rewards_telescope():
    """J must equal cash plus the marked inventory minus the penalties, exactly."""
    setting = equity_setting()
    crn = make_crn(setting, 200, seed=7)
    rollout = simulate(setting, ConstantPolicy(setting, 0.4), crn)
    rebuilt = (rollout.cash + rollout.terminal_q * rollout.terminal_s
               - rollout.penalty - setting.a * rollout.terminal_q ** 2)
    np.testing.assert_allclose(rollout.pnl, rebuilt, rtol=1e-10, atol=1e-8)


def test_inventory_never_leaves_the_bound_and_fills_match_the_arrival_rate():
    """Quoting at a zero depth is the most aggressive possible quote, so the fill count must
    match the arrival intensity, and the inventory must still respect |q| <= Q."""
    setting = credit_setting()
    crn = make_crn(setting, 200, seed=8)
    rollout = simulate(setting, ConstantPolicy(setting, 0.0), crn)
    assert np.abs(rollout.terminal_q).max() <= setting.Q
    assert rollout.steps_at_bound.max() > 0
    expected = 2.0 * setting.lam * setting.fill_model().f(0.0) * setting.T
    observed = (rollout.ask_fills + rollout.bid_fills).mean()
    assert 0.7 * expected < observed < 1.3 * expected, (
        f'observed {observed:.2f} fills per episode, arrival intensity predicts {expected:.2f}'
    )


def test_deeper_quotes_fill_less_often():
    setting = equity_setting()
    crn = make_crn(setting, 500, seed=9)
    shallow = simulate(setting, ConstantPolicy(setting, 0.1), crn)
    deep = simulate(setting, ConstantPolicy(setting, 1.0), crn)
    assert shallow.ask_fills.mean() > deep.ask_fills.mean()
    assert shallow.bid_fills.mean() > deep.bid_fills.mean()


def test_crn_is_reproducible_and_shared():
    setting = equity_setting()
    a = make_crn(setting, 50, seed=11)
    b = make_crn(setting, 50, seed=11)
    c = make_crn(setting, 50, seed=12)
    assert np.array_equal(a.price_shocks, b.price_shocks)
    assert not np.array_equal(a.price_shocks, c.price_shocks)


@pytest.mark.parametrize('setter', [equity_setting, credit_setting])
def test_dp_value_matches_monte_carlo_of_the_dp_policy(setter):
    """The dynamic program claims the exact expected objective of the discrete simulator, so
    simulating its own policy must reproduce that number within sampling error."""
    setting = setter()
    dp = solve_dp(setting)
    rollout = simulate(setting, dp_policy(dp, setting), make_crn(setting, 4000, seed=101))
    gap = rollout.pnl.mean() - dp.optimal_value
    assert abs(gap) < 4.0 * _se(rollout), (
        f'{setting.name}: MC mean {rollout.pnl.mean():.4f} vs DP {dp.optimal_value:.4f}, '
        f'gap {gap:.4f} with standard error {_se(rollout):.4f}'
    )


@pytest.mark.parametrize('setter', [equity_setting, credit_setting])
def test_no_policy_beats_the_dynamic_program(setter):
    """Every reference policy must come in at or below the exact optimum, within noise."""
    setting = setter()
    dp = solve_dp(setting)
    crn = make_crn(setting, 4000, seed=202)
    policies = [hjb_policy(setting), ConstantPolicy(setting, 0.1),
                ConstantPolicy(setting, 0.5), ConstantPolicy(setting, 1.0)]
    if setting.fill_kind == 'exponential':
        policies.append(ASHeuristicPolicy(setting))
    for policy in policies:
        rollout = simulate(setting, policy, crn)
        assert rollout.pnl.mean() < dp.optimal_value + 4.0 * _se(rollout), (
            f'{setting.name}: policy {policy.name} beat the dynamic program'
        )


def test_hjb_policy_is_close_to_the_discrete_optimum():
    """The continuous-time solution is a O(dt) approximation of the discrete optimum, so the
    two must be close but the discrete optimum must not be below it."""
    setting = equity_setting()
    dp = solve_dp(setting)
    crn = make_crn(setting, 4000, seed=303)
    hjb = simulate(setting, hjb_policy(setting), crn)
    dpv = simulate(setting, dp_policy(dp, setting), crn)
    assert dp.optimal_value >= hjb.pnl.mean() - 4.0 * _se(hjb)
    assert dp.optimal_value - hjb.pnl.mean() < 0.2 * abs(dp.optimal_value), (
        f'gap {dp.optimal_value - hjb.pnl.mean():.4f} too large relative to the optimum'
    )
    assert (dpv.pnl - hjb.pnl).mean() > -4.0 * (dpv.pnl - hjb.pnl).std(ddof=1) / np.sqrt(4000)


def test_price_shocks_move_dispersion_but_not_the_mean():
    """The mid price is a martingale that is independent of the fills, so switching the price
    shocks off must leave the expected objective and the optimal policy value unchanged."""
    setting = equity_setting()
    no_price = replace(setting, sigma=0.0)
    dp = solve_dp(setting)
    dp_flat = solve_dp(no_price)
    assert dp_flat.optimal_value == pytest.approx(dp.optimal_value, rel=1e-9)
    np.testing.assert_allclose(dp.d_ask, dp_flat.d_ask, atol=1e-9, equal_nan=True)
    noisy = simulate(setting, dp_policy(dp, setting), make_crn(setting, 2000, seed=404))
    flat = simulate(no_price, dp_policy(dp_flat, no_price), make_crn(no_price, 2000, seed=404))
    assert flat.pnl.var(ddof=1) < 0.8 * noisy.pnl.var(ddof=1)
    assert abs(noisy.pnl.mean() - flat.pnl.mean()) < 4.0 * np.hypot(_se(noisy), _se(flat))


def _brute_force_step(setting, value_next, step, q_index, n_grid=801):
    """Objective of one Bellman step by exhaustive 2-D search over the depth grid."""
    dt = setting.dt
    fill = setting.fill_model()
    q_grid = np.arange(-setting.Q, setting.Q + 1)
    q = float(q_grid[q_index])
    v_here = value_next[q_index]
    v_up = value_next[q_index + 1] if q_index + 1 < q_grid.size else v_here
    v_down = value_next[q_index - 1] if q_index > 0 else v_here
    curvature = 2.0 * v_here - v_up - v_down
    d = np.linspace(0.0, setting.d_max, n_grid)
    p = 1.0 - np.exp(-setting.lam * fill.f(d) * dt)
    if q_index == 0:
        p = np.zeros_like(p)
    if q_index == q_grid.size - 1:
        p_b = np.zeros_like(p)
    else:
        p_b = 1.0 - np.exp(-setting.lam * fill.f(d) * dt)
    obj = (p[:, None] * d[:, None] + p_b[None, :] * d[None, :]
           + v_here
           + p_b[None, :] * (v_up - v_here)
           + p[:, None] * (v_down - v_here)
           + p[:, None] * p_b[None, :] * curvature
           - setting.phi * q ** 2 * dt)
    flat = int(np.argmax(obj))
    # axis 0 of obj is the ask depth, axis 1 is the bid depth
    return obj.max(), d[flat // n_grid], d[flat % n_grid]


@pytest.mark.parametrize('setter', [equity_setting, credit_setting])
def test_dp_step_optimum_matches_brute_force_2d_search(setter):
    """Coordinate ascent must find the true joint optimum of a Bellman step."""
    setting = setter()
    dp = solve_dp(setting)
    steps = [0, 1, setting.n_steps // 3, setting.n_steps // 2, setting.n_steps - 2]
    q_indices = [0, 1, setting.Q - 1, setting.Q, setting.Q + 1, 2 * setting.Q - 1,
                 2 * setting.Q]
    for step in steps:
        for q_index in q_indices:
            best, d_ask, d_bid = _brute_force_step(setting, dp.value[step + 1], step, q_index)
            assert dp.value[step, q_index] == pytest.approx(best, abs=5e-4), (
                f'{setting.name} step {step} q {dp.q[q_index]}: DP {dp.value[step, q_index]:.6f} '
                f'vs brute force {best:.6f}'
            )
            for brute, tabulated, side in ((d_ask, dp.d_ask[step, q_index], 'ask'),
                                           (d_bid, dp.d_bid[step, q_index], 'bid')):
                if not np.isfinite(tabulated):
                    continue
                assert abs(brute - tabulated) < 2.0 * setting.d_max / 800, (
                    f'{setting.name} step {step} q {dp.q[q_index]} {side}: DP depth '
                    f'{tabulated:.5f} vs brute force {brute:.5f}'
                )
