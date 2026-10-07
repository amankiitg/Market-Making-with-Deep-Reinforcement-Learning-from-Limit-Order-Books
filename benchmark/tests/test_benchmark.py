"""Pipeline level tests: is the benchmark well posed, and does the harness measure what it
claims to measure.

Everything here is cheap except the RL tests, which are marked slow and train a short budget.
"""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmark.config import LOGIT_SCALE, credit_setting, equity_setting
from benchmark.dp import solve_dp
from benchmark.evaluate import (efficiency, evaluate, paired_difference, held_out_crn,
                               tune_constant_depth)
from benchmark.simulator import make_crn, simulate
from benchmark.strategies import ConstantPolicy, dp_policy, hjb_policy

SETTERS = [equity_setting, credit_setting]


def _naive_grid(setting):
    return np.linspace(0.02 * setting.d_max, setting.d_max, 13)


def test_training_cannot_see_the_optimum():
    """The trainer and the environment must not import or name the solver or the dynamic
    program. The ground truth is allowed in evaluation, reporting and tests only."""
    banned_modules = ('benchmark.dp', 'benchmark.hjb')
    banned_names = ('solve_dp', 'solve_hjb', 'optimal_value', 'closed_form_exponential',
                    'j_opt', 'j_star', 'dsolution')
    for relative in ('benchmark/train.py', 'benchmark/envs.py'):
        source = Path(relative).read_text()
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        offending = [m for m in modules if m.startswith(banned_modules)]
        assert not offending, f'{relative} imports {offending}'
        found = [name for name in banned_names if name in source]
        assert not found, f'{relative} mentions {found}, which belongs to evaluation only'


def test_credit_calibration_hits_its_stated_targets():
    """The credit market is calibrated in spread terms: a 2.5 basis point half-spread and a
    30 percent hit ratio at (t = 0, q = 0), both as solved by the dynamic program."""
    setting = credit_setting()
    dp = solve_dp(setting)
    d0 = float(dp.d_ask[0, setting.Q])
    hit = float(setting.fill_model().f(d0))
    assert d0 == pytest.approx(0.175, rel=0.20), f'half-spread at (0,0) is {d0:.4f} points'
    assert hit == pytest.approx(0.30, abs=0.05), f'hit ratio at the optimum is {hit:.3f}'
    # 1 basis point of spread is 0.07 price points at a DV01 of 700 USD per lot
    assert d0 / 0.07 == pytest.approx(2.5, rel=0.20)
    assert setting.sigma == pytest.approx(4 * 0.07)
    assert 0.5 * setting.gamma * setting.sigma ** 2 * setting.T == pytest.approx(2.25 * d0,
                                                                               rel=0.05)


def test_credit_inventory_limit_and_action_bound_rules():
    """Q must never bind and the action bound must cost under 0.1 percent of the optimum while
    leaving the optimal depth between 20 and 50 percent of the range."""
    setting = credit_setting()
    dp = solve_dp(setting)
    d0 = float(dp.d_ask[0, setting.Q])
    assert 0.20 <= d0 / setting.d_max <= 0.50, f'depth is {d0 / setting.d_max:.3f} of the range'
    wide = solve_dp(replace(setting, d_max=4.0 * setting.d_max))
    loss = wide.optimal_value - dp.optimal_value
    assert loss >= 0.0
    assert loss < 1e-3 * abs(wide.optimal_value), (
        f'the bound costs {100 * loss / abs(wide.optimal_value):.4f} percent of the optimum'
    )
    crn = held_out_crn(setting, n_episodes=400)
    from benchmark.strategies import dp_policy
    rollout = simulate(setting, dp_policy(dp, setting), crn)
    assert np.abs(rollout.terminal_q).max() < setting.Q, 'the optimal policy must not reach Q'


@pytest.mark.parametrize('setter', SETTERS)
def test_the_action_bound_is_almost_free(setter):
    """The RL action bound must not be the binding constraint on the answer: solving the same
    problem with a bound four times wider must change the optimum by a negligible amount."""
    setting = setter()
    wide = replace(setting, d_max=4.0 * setting.d_max)
    j_bound = solve_dp(setting).optimal_value
    j_wide = solve_dp(wide).optimal_value
    loss = j_wide - j_bound
    assert loss >= 0.0, 'a wider action set cannot be worse'
    assert loss < 1e-3 * abs(j_wide), (
        f'{setting.name}: the bound costs {loss:.6f} of {j_wide:.6f} '
        f'({100 * loss / abs(j_wide):.4f} percent)'
    )


@pytest.mark.parametrize('setter', SETTERS)
def test_a_constant_quote_is_clearly_suboptimal(setter):
    """The benchmark is only informative if beating a tuned constant quote is worth something.
    If the gap were tiny, the efficiency score would be dominated by Monte Carlo noise."""
    setting = setter()
    j_opt = solve_dp(setting).optimal_value
    depth, rows = tune_constant_depth(setting, _naive_grid(setting))
    j_naive = max(row['mean_pnl'] for row in rows)
    assert depth > 0.0
    gap = j_opt - j_naive
    assert gap > 0.2 * abs(j_opt), (
        f'{setting.name}: tuned constant depth {depth:.4f} reaches {j_naive:.4f} against an '
        f'optimum of {j_opt:.4f}, a gap of only {gap:.4f}'
    )


@pytest.mark.parametrize('setter', SETTERS)
def test_efficiency_of_the_exact_policies_is_one(setter):
    """The optimum and the continuous time solution must both land at an efficiency of one,
    within sampling error. This is the end to end calibration check of the headline metric."""
    setting = setter()
    crn = held_out_crn(setting, n_episodes=3000)
    naive_depth, _ = tune_constant_depth(setting, _naive_grid(setting))
    policies = {'dp_optimal': dp_policy(solve_dp(setting), setting),
                'hjb_optimal': hjb_policy(setting),
                'naive_tuned': ConstantPolicy(setting, naive_depth)}
    reports = evaluate(setting, policies, crn)
    j_opt = solve_dp(setting).optimal_value
    j_naive = reports['naive_tuned'].mean
    for name in ('dp_optimal', 'hjb_optimal'):
        score = efficiency(reports[name].mean, j_naive, j_opt)
        assert abs(score - 1.0) < 0.15, (
            f'{setting.name}: {name} has efficiency {score:.3f}, expected one within noise'
        )


@pytest.mark.parametrize('setter', SETTERS)
def test_environment_matches_the_vectorised_simulator(setter):
    """Two independent implementations of the same process must agree on the mean objective."""
    pytest.importorskip('gymnasium')
    from benchmark.envs import MMBenchEnv

    setting = setter()
    depth = float(solve_dp(setting).d_ask[0, setting.Q])
    logit = float(np.log(depth / (setting.d_max - depth)) / LOGIT_SCALE)
    n_episodes = 1200
    model_free = simulate(setting, ConstantPolicy(setting, depth),
                          make_crn(setting, n_episodes, seed=555)).pnl
    env = MMBenchEnv(setting, seed=556, noise_free=False)
    returns = []
    for _ in range(n_episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, done, _, _ = env.step(np.array([logit, logit], dtype=np.float32))
            total += reward
        returns.append(total)
    returns = np.array(returns)
    se = np.hypot(returns.std(ddof=1), model_free.std(ddof=1)) / np.sqrt(n_episodes)
    assert abs(returns.mean() - model_free.mean()) < 3.0 * se, (
        f'{setting.name}: environment mean {returns.mean():.4f} vs simulator mean '
        f'{model_free.mean():.4f}, difference {returns.mean() - model_free.mean():.4f} '
        f'with standard error {se:.4f}'
    )


@pytest.mark.parametrize('setter', SETTERS)
def test_environment_passes_the_gymnasium_checker(setter):
    pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env

    from benchmark.envs import MMBenchEnv

    check_env(MMBenchEnv(setter(), seed=1), skip_render_check=True)


def test_paired_difference_detects_a_known_offset():
    setting = equity_setting()
    crn = held_out_crn(setting, n_episodes=500)
    a = evaluate(setting, {'a': ConstantPolicy(setting, 0.7)}, crn)['a']
    b = evaluate(setting, {'b': ConstantPolicy(setting, 0.7)}, crn)['b']
    assert paired_difference(a, b)['mean_difference'] == pytest.approx(0.0, abs=1e-12)
    worse = evaluate(setting, {'c': ConstantPolicy(setting, setting.d_max)}, crn)['c']
    result = paired_difference(a, worse)
    assert result['mean_difference'] > 0.0
    assert result['p_value'] < 0.01


@pytest.mark.slow
@pytest.mark.parametrize('setter,timesteps,threshold', [
    (equity_setting, 100_000, 0.60),
    (credit_setting, 300_000, 0.35),
])
def test_rl_reaches_a_large_share_of_the_optimum(setter, timesteps, threshold):
    """A short training run must capture a substantial share of the distance from a deep fixed
    quote to the exact optimum. The thresholds are about 70 percent of the 5-seed mean that
    benchmark/run_all.py measured at 500k steps (0.910 for equity, 0.566 for credit), rounded
    down to the nearest 0.05, and the budgets here are shorter than the full run, so the point
    is to catch a broken environment, reward or action mapping, not to measure final
    performance."""
    from benchmark.strategies import RLPolicy
    from benchmark.train import TrainingConfig, train_ppo

    setting = setter()
    j_opt = solve_dp(setting).optimal_value
    crn = held_out_crn(setting, n_episodes=1000)
    deep = evaluate(setting, {'constant_deep': ConstantPolicy(setting, setting.d_max)}, crn)
    config = TrainingConfig(total_timesteps=timesteps, eval_every=timesteps)
    run = train_ppo(setting, 0, eval_crn=crn, config=config)
    rl = evaluate(setting, {'rl': RLPolicy(run.model, setting)}, crn)['rl']
    score = efficiency(rl.mean, deep['constant_deep'].mean, j_opt)
    assert score > threshold, (
        f'{setting.name}: after {timesteps} steps the agent captures only {score:.2f} of the '
        f'distance from a deep fixed quote to the optimum'
    )
