"""Evaluation harness: baselines, held out episodes, metrics, and the efficiency score.

Discipline that the numbers depend on
-------------------------------------
* Tuning and testing use disjoint episode sets. The tuned constant depth baseline is chosen
  on tuning episodes (seeds `TUNING_SEEDS`) and every headline number is then produced on the
  held out test episodes of one common random number set.
* Every strategy is evaluated on the same price shocks and fill uniforms, so differences
  between strategies are paired and far more precisely estimated than the individual means.
* E_J = (J_strategy - J_naive) / (J_opt - J_naive) is the fraction of the gap between a tuned
  constant-depth baseline and the exact optimum that a strategy closes. The optimum comes from
  the dynamic program, which is exact for the discretised simulator.
* The RL agent must not beat the optimum by more than sampling error. `paired_vs_optimum`
  reports the paired difference against the optimal policy together with its standard error so
  that any such violation is visible rather than averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from benchmark.config import Setting
from benchmark.simulator import CRN, Rollout, Policy, make_crn, simulate

TUNING_SEEDS = (1001, 1002, 1003)
TEST_SEED = 999
N_TEST_EPISODES = 2000
N_TUNING_EPISODES = 400


@dataclass
class StrategyReport:
    name: str
    rollout: Rollout

    @property
    def pnl(self) -> np.ndarray:
        return self.rollout.pnl

    @property
    def mean(self) -> float:
        return float(self.rollout.pnl.mean())

    @property
    def se(self) -> float:
        return float(self.rollout.pnl.std(ddof=1) / np.sqrt(self.rollout.n_episodes))

    def summary(self) -> dict:
        out = self.rollout.summary()
        out['strategy'] = self.name
        return out


def held_out_crn(setting: Setting, n_episodes: int = N_TEST_EPISODES, seed: int = TEST_SEED) -> CRN:
    return make_crn(setting, n_episodes, seed)


def tuning_crn(setting: Setting, seed: int) -> CRN:
    return make_crn(setting, N_TUNING_EPISODES, seed)


def tune_constant_depth(setting: Setting, depths: np.ndarray) -> tuple[float, list[dict]]:
    """Pick the fixed depth that maximises the mean objective on the tuning episodes."""
    from benchmark.strategies import ConstantPolicy

    crns = [tuning_crn(setting, seed) for seed in TUNING_SEEDS]
    rows = []
    best_depth, best_value = float(depths[0]), -np.inf
    for depth in depths:
        policy = ConstantPolicy(setting, float(depth))
        values = [simulate(setting, policy, crn).pnl.mean() for crn in crns]
        value = float(np.mean(values))
        rows.append({'depth': float(depth), 'mean_pnl': value,
                     'spread_across_tuning_seeds': float(np.std(values, ddof=1))})
        if value > best_value:
            best_depth, best_value = float(depth), value
    return best_depth, rows


def evaluate(setting: Setting, policies: dict[str, Policy], crn: CRN) -> dict[str, StrategyReport]:
    return {name: StrategyReport(name=name, rollout=simulate(setting, policy, crn))
            for name, policy in policies.items()}


def efficiency(j_strategy: float, j_naive: float, j_opt: float) -> float:
    denominator = j_opt - j_naive
    if abs(denominator) < 1e-12:  # pragma: no cover - defensive
        return float('nan')
    return (j_strategy - j_naive) / denominator


def seed_dispersion(per_seed_objectives, j_naive: float, j_opt: float) -> dict:
    """Efficiency per training seed, with the spread across seeds.

    Every seed is evaluated on the same held out episodes, so the spread is training variation
    rather than evaluation noise. The worst and best seeds are reported alongside the mean
    because a single lucky seed is not a result.
    """
    objectives = [float(v) for v in per_seed_objectives]
    efficiencies = np.asarray([efficiency(v, j_naive, j_opt) for v in objectives], dtype=float)
    return {'per_seed_objective': objectives,
            'per_seed_efficiency': efficiencies.tolist(),
            'mean_efficiency': float(efficiencies.mean()),
            'std_efficiency': float(efficiencies.std(ddof=1)) if efficiencies.size > 1 else 0.0,
            'worst_seed_efficiency': float(efficiencies.min()),
            'best_seed_efficiency': float(efficiencies.max()),
            'n_seeds': int(efficiencies.size)}


def headline_sentence(dispersion: dict) -> str:
    """The one line summary in the format the project brief asks for."""
    return (f"PPO recovers {100 * dispersion['mean_efficiency']:.1f}% "
            f"+/- {100 * dispersion['std_efficiency']:.1f}% "
            f"(worst seed {100 * dispersion['worst_seed_efficiency']:.1f}%) of the optimal "
            f"policy's improvement over a tuned constant quote.")


def tuned_quote_earns_nothing(reports: dict, naive_name: str = 'naive_tuned') -> bool:
    """True when the tuned constant quote's mean is within its own 95 percent interval of zero."""
    report = reports[naive_name]
    half_width = 1.959963984540054 * report.se
    return abs(report.mean) <= half_width


def paired_difference(a: StrategyReport, b: StrategyReport) -> dict:
    """Paired comparison of two strategies evaluated on the same episodes."""
    diff = a.pnl - b.pnl
    n = diff.size
    se = float(diff.std(ddof=1) / np.sqrt(n))
    t_stat = float(diff.mean() / se) if se > 0 else float('nan')
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1)) if se > 0 else float('nan')
    return {'comparing': f'{a.name} - {b.name}',
            'mean_difference': float(diff.mean()),
            'standard_error': se,
            'ci95_low': float(diff.mean() - 1.959963984540054 * se),
            'ci95_high': float(diff.mean() + 1.959963984540054 * se),
            't_statistic': t_stat,
            'p_value': p_value}


def build_summary(setting: Setting, reports: dict[str, StrategyReport], j_opt: float,
                  naive_name: str, pnl_scale: float) -> dict:
    """Headline table: every strategy against the exact optimum and the tuned baseline."""
    naive = reports[naive_name].mean
    rows = []
    for name, report in reports.items():
        s = report.summary()
        rows.append({
            'strategy': name,
            'mean_pnl': s['mean_pnl'],
            'mean_pnl_reported': s['mean_pnl'] * pnl_scale,
            'ci95_half_width_reported': s['ci95_half_width'] * pnl_scale,
            'std_pnl_reported': s['std_pnl'] * pnl_scale,
            'pnl_over_std': s['mean_pnl'] / s['std_pnl'] if s['std_pnl'] > 0 else float('nan'),
            'efficiency': efficiency(s['mean_pnl'], naive, j_opt),
            'mean_abs_terminal_q': s['mean_abs_terminal_q'],
            'fills_per_episode': s['asks_filled_per_episode'] + s['bids_filled_per_episode'],
            'fraction_steps_at_bound': s['steps_at_bound_fraction'],
            'mean_depth_ask': s['mean_depth_ask'],
            'mean_depth_bid': s['mean_depth_bid'],
        })
    return {'setting': setting.name,
            'j_opt_discrete': j_opt,
            'j_naive_tuned': naive,
            'j_naive_ci95_half_width': 1.959963984540054 * reports[naive_name].se,
            'pnl_scale': pnl_scale,
            'rows': rows,
            'paired': [paired_difference(reports[name], reports['dp_optimal'])
                       for name in reports if name != 'dp_optimal']}


def format_summary_table(summary: dict, unit_label: str) -> str:
    lines = [f"| strategy | mean PnL ({unit_label}) | 95% CI | PnL / std | efficiency vs optimum | "
             f"mean abs terminal q | fills / day | time at bound |",
             '|---|---|---|---|---|---|---|---|']
    for row in summary['rows']:
        ci = row['ci95_half_width_reported']
        lines.append(
            f"| {row['strategy']} | {row['mean_pnl_reported']:.2f} | +-{ci:.2f} | "
            f"{row['pnl_over_std']:.2f} | {row['efficiency']:.3f} | "
            f"{row['mean_abs_terminal_q']:.3f} | {row['fills_per_episode']:.1f} | "
            f"{row['fraction_steps_at_bound']:.4f} |")
    return '\n'.join(lines)
