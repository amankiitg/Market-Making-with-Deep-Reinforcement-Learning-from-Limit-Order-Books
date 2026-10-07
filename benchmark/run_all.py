"""Run the whole benchmark: solve, train, evaluate, plot, and print the summary.

    python -m benchmark.run_all                # full run, about 15 minutes on a MacBook CPU
    python -m benchmark.run_all --quick        # smoke run, about 1 minute

Everything is written to benchmark/results/ (JSON plus a markdown summary) and figures/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.config import Setting, credit_setting, equity_setting, parameter_table
from benchmark.dp import solve_dp
from benchmark.evaluate import (StrategyReport, build_summary, evaluate, format_summary_table,
                               held_out_crn, tune_constant_depth)
from benchmark.figures import (plot_efficiency_bars, plot_inventory_histograms,
                               plot_learning_curves, plot_policy_depths, plot_sample_path)
from benchmark.strategies import (ASHeuristicPolicy, ConstantPolicy, RLPolicy,
                                 TablePolicy, closed_form_policy, dp_policy, hjb_policy)

RESULTS = Path(__file__).resolve().parent / 'results'
RL_SEEDS = (0, 1, 2, 3, 4)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def naive_depth_grid(setting: Setting) -> np.ndarray:
    """A depth grid around the region that matters, from a shallow quote to a wide one."""
    return np.round(np.linspace(0.05 * setting.d_max, 1.0 * setting.d_max, 21), 6) \
        if setting.name == 'credit' else np.round(np.linspace(0.1, 2.0, 20), 4)


def build_policies(setting: Setting, dp, hjb: TablePolicy, naive_depth: float) -> dict:
    policies = {
        'dp_optimal': dp_policy(dp, setting),
        'hjb_optimal': hjb,
        'naive_tuned': ConstantPolicy(setting, naive_depth, name='naive_tuned'),
        'constant_deep': ConstantPolicy(setting, setting.d_max, name='constant_deep'),
    }
    if setting.fill_kind == 'exponential':
        policies['closed_form_cartea_jaimungal'] = closed_form_policy(setting)
        policies['avellaneda_stoikov'] = ASHeuristicPolicy(setting)
    return policies


def run_setting(setting: Setting, timesteps: int, seeds, n_test: int, train: bool,
                noise_free: bool = True, figures: bool = True, verbose: bool = True,
                figure_tag: str = '') -> dict:
    started = time.time()
    dp = solve_dp(setting)
    j_opt = dp.optimal_value
    hjb = hjb_policy(setting)
    naive_depth, tuning_rows = tune_constant_depth(setting, naive_depth_grid(setting))
    crn = held_out_crn(setting, n_episodes=n_test)
    policies = build_policies(setting, dp, hjb, naive_depth)
    reports = evaluate(setting, policies, crn)

    runs = []
    if train:
        from benchmark.train import TrainingConfig, train_ppo

        config = TrainingConfig(total_timesteps=timesteps, noise_free=noise_free)
        for seed in seeds:
            run = train_ppo(setting, seed, j_opt, eval_crn=crn, config=config)
            runs.append(run)
            if verbose:
                print(f'  [{setting.name}] seed {seed}: {run.wall_time:5.1f}s, '
                      f'held out objective {run.history[-1][1]:.4f} '
                      f'(optimum {j_opt:.4f})', flush=True)
        per_seed = [evaluate(setting, {'rl': RLPolicy(run.model, setting)}, crn)['rl']
                    for run in runs]
        # the RL entry is the mixture over seeds, so its per-episode values are the seed mean
        stacked = np.mean([report.pnl for report in per_seed], axis=0)
        reports['rl'] = StrategyReport(name='rl',
                                       rollout=_with_pnl(per_seed[0].rollout, stacked))
        if len(runs) > 1:
            reports['rl_worst_seed'] = StrategyReport(
                name='rl_worst_seed', rollout=min(per_seed, key=lambda r: r.pnl.mean()).rollout)

    summary = build_summary(setting, reports, j_opt, naive_name='naive_tuned',
                            pnl_scale=setting.usd_per_price_unit)
    summary['naive_depth'] = naive_depth
    summary['tuning'] = tuning_rows
    summary['hjb_value_continuous_time'] = continuous_time_value(setting)
    summary['wall_time_seconds'] = time.time() - started
    summary['rl_seed_objectives'] = [float(r.history[-1][1]) for r in runs]
    summary['rl_seed_wall_times'] = [float(r.wall_time) for r in runs]
    summary['rl_histories'] = [[[int(s), float(v)] for s, v in r.history] for r in runs]

    if figures:
        paths = []
        paths.append(str(plot_policy_depths(setting, dp, hjb, tag=figure_tag)))
        shown = ['dp_optimal', 'hjb_optimal', 'rl', 'naive_tuned']
        paths.append(str(plot_inventory_histograms(
            setting, reports, [n for n in shown if n in reports], tag=figure_tag)))
        if runs:
            paths.append(str(plot_learning_curves(
                setting, [r.history for r in runs], j_opt,
                reports['naive_tuned'].mean, reports['hjb_optimal'].mean, tag=figure_tag)))
        showcased = RLPolicy(runs[0].model, setting) if runs else policies['dp_optimal']
        paths.append(str(plot_sample_path(setting, showcased, crn, tag=figure_tag)))
        summary['figures'] = paths
    return summary


def continuous_time_value(setting: Setting) -> float:
    """h(0, 0) of the continuous time solution, for reference next to the discrete optimum."""
    from benchmark.hjb import solve_hjb

    result = solve_hjb(setting.fill_model(), T=setting.T, Q=setting.Q, phi=setting.phi,
                       a=setting.a, lam=setting.lam,
                       t_eval=np.linspace(0.0, setting.T, 51))
    return float(result.h[0, setting.Q])


def _with_pnl(rollout, pnl):
    import dataclasses

    return dataclasses.replace(rollout, pnl=pnl)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=300_000)
    parser.add_argument('--seeds', type=int, default=len(RL_SEEDS))
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='fast smoke run: few steps, few episodes, one seed')
    parser.add_argument('--ablation', action='store_true',
                        help='also train on the fully realised PnL in both settings, which '
                             'quantifies what the variance reduction in benchmark/envs.py is '
                             'worth')
    args = parser.parse_args(argv)

    timesteps = 2_000 if args.quick else args.timesteps
    seeds = RL_SEEDS[:1] if args.quick else RL_SEEDS[:args.seeds]
    episodes = 200 if args.quick else args.episodes

    summaries = []
    main_summaries = []
    for setter in (equity_setting, credit_setting):
        setting = setter()
        print(f'=== {setting.name}: {setting.market}', flush=True)
        summary = run_setting(setting, timesteps=timesteps, seeds=seeds, n_test=episodes,
                              train=not args.no_train,
                              figure_tag='_quick' if args.quick else '')
        summaries.append(summary)
        main_summaries.append(summary)
        unit = 'USD per 100-share lot per day' if setting.name == 'equity' \
            else 'USD per 1mm face lot per day'
        print(f'    exact discrete optimum J* = {summary["j_opt_discrete"]:.4f} raw units, '
              f'tuned constant depth = {summary["naive_depth"]:.4f}')
        print(format_summary_table(summary, unit))
        print('    paired against the optimum:')
        for row in summary['paired']:
            print(f'      {row["comparing"]:34s} {row["mean_difference"]:9.4f} '
                  f'+- {1.959963984540054 * row["standard_error"]:.4f} '
                  f'(p = {row["p_value"]:.3f})')
        print(f'    wall time {summary["wall_time_seconds"]:.1f}s')

    if args.ablation:
        for setter in (equity_setting, credit_setting):
            setting = setter()
            print(f'=== {setting.name} ablation: training on the fully realised PnL instead of '
                  f'the variance reduced reward', flush=True)
            summary = run_setting(setting, timesteps=timesteps, seeds=seeds, n_test=episodes,
                                  train=True, noise_free=False, figures=False)
            summary['setting'] = f'{setting.name}_realised_pnl_reward'
            summaries.append(summary)
            unit = 'USD per 100-share lot per day' if setting.name == 'equity' \
                else 'USD per 1mm face lot per day'
            print(format_summary_table(summary, unit), flush=True)

    if len(main_summaries) > 1:
        plot_efficiency_bars(main_summaries, tag='_quick' if args.quick else '')
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = jsonable({'summaries': summaries,
                        'parameters': {s.name: parameter_table(s)
                                       for s in (equity_setting(), credit_setting())}})
    out = RESULTS / ('summary_quick.json' if args.quick else 'summary.json')
    out.write_text(json.dumps(payload, indent=2))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
