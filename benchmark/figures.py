"""Figures for the benchmark. Each function writes one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from benchmark.config import Setting       # noqa: E402
from benchmark.dp import DPResult          # noqa: E402
from benchmark.strategies import TablePolicy  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / 'figures'


def _out(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name


def plot_policy_depths(setting: Setting, dp: DPResult, hjb: TablePolicy,
                       times=(0.0, 0.5, 0.9), tag: str = '') -> Path:
    """Optimal depths against inventory at several times of day, solver against dynamic program."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    q = dp.q
    for frac in times:
        index = int(round(frac * setting.n_steps))
        axes[0].plot(q, dp.d_ask[index], marker='o', ms=2.5, label=f'DP, t/T={frac:.1f}')
        axes[0].plot(q, hjb.d_ask[index], ls='--', lw=1.0,
                     label=f'HJB, t/T={frac:.1f}')
        axes[1].plot(q, dp.d_bid[index], marker='o', ms=2.5)
        axes[1].plot(q, hjb.d_bid[index], ls='--', lw=1.0)
    axes[0].set_title('ask depth')
    axes[1].set_title('bid depth')
    for ax in axes:
        ax.set_xlabel('inventory q (lots)')
        ax.set_ylabel('depth from mid (price units)')
        ax.grid(alpha=0.3)
        ax.axvline(0.0, color='k', lw=0.6, alpha=0.4)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'Optimal quote depths, {setting.name} setting')
    fig.tight_layout()
    path = _out(f'depths_{setting.name}{tag}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_inventory_histograms(setting: Setting, reports: dict, names,
                              tag: str = '') -> Path:
    """Terminal inventory distribution per strategy on the held out episodes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-setting.Q - 0.5, setting.Q + 1.5, 1.0)
    for name in names:
        ax.hist(reports[name].rollout.terminal_q, bins=bins, histtype='step', density=True,
                label=name, lw=1.4)
    ax.set_xlabel('terminal inventory (lots)')
    ax.set_ylabel('density')
    ax.set_title(f'End of day inventory, {setting.name} setting')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _out(f'inventory_{setting.name}{tag}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_learning_curves(setting: Setting, histories: list, j_opt: float, j_naive: float,
                         j_hjb: float, tag: str = '') -> Path:
    """Mean and range of the held out objective across training seeds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.array([step for step, _ in histories[0]])
    values = np.array([[value for _, value in history] for history in histories])
    ax.plot(steps, values.mean(axis=0), lw=1.8, label='RL, mean of seeds')
    ax.fill_between(steps, values.min(axis=0), values.max(axis=0), alpha=0.2,
                    label='RL, range of seeds')
    ax.axhline(j_opt, color='k', ls='-', lw=1.0, label='discrete optimum (DP)')
    ax.axhline(j_hjb, color='tab:green', ls=':', lw=1.2, label='HJB optimal policy')
    ax.axhline(j_naive, color='tab:red', ls='--', lw=1.0, label='tuned constant depth')
    ax.set_xlabel('environment steps')
    ax.set_ylabel('mean objective per episode')
    ax.set_title(f'Learning curves, {setting.name} setting')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _out(f'learning_curves_{setting.name}{tag}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sample_path(setting: Setting, policy, crn, episode: int = 0,
                     tag: str = '') -> Path:
    """One episode: the quotes the policy posts, the resulting inventory, and the fills."""
    single = type(crn)(price_shocks=crn.price_shocks[:, episode:episode + 1],
                       fill_ask=crn.fill_ask[:, episode:episode + 1],
                       fill_bid=crn.fill_bid[:, episode:episode + 1], seed=crn.seed)
    # re-run the episode recording the quotes
    dt = setting.dt
    fill = setting.fill_model()
    q = 0
    mid = setting.s0
    asks, bids, mids, invs = [], [], [], []
    for step in range(setting.n_steps):
        d_ask, d_bid = policy(np.array([step]), np.array([q]))
        d_ask = float(np.clip(np.asarray(d_ask).ravel()[0], 0.0, setting.d_max))
        d_bid = float(np.clip(np.asarray(d_bid).ravel()[0], 0.0, setting.d_max))
        p_ask = 1.0 - np.exp(-setting.lam * fill.f(d_ask) * dt)
        p_bid = 1.0 - np.exp(-setting.lam * fill.f(d_bid) * dt)
        if q <= -setting.Q:
            p_ask = 0.0
        if q >= setting.Q:
            p_bid = 0.0
        q += int(single.fill_bid[step, 0] < p_bid) - int(single.fill_ask[step, 0] < p_ask)
        mid += setting.sigma * np.sqrt(dt) * single.price_shocks[step, 0]
        asks.append(mid + d_ask)
        bids.append(mid - d_bid)
        mids.append(mid)
        invs.append(q)
    steps = np.arange(setting.n_steps) * dt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(steps, mids, color='k', lw=1.0, label='mid')
    axes[0].plot(steps, asks, color='tab:red', lw=1.0, label='ask quote')
    axes[0].plot(steps, bids, color='tab:blue', lw=1.0, label='bid quote')
    axes[0].set_ylabel('price')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].step(steps, invs, where='post', color='tab:purple', lw=1.2)
    axes[1].axhline(setting.Q, color='k', lw=0.6, ls=':')
    axes[1].axhline(-setting.Q, color='k', lw=0.6, ls=':')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('inventory (lots)')
    axes[1].grid(alpha=0.3)
    fig.suptitle(f'Sample episode, {setting.name} setting')
    fig.tight_layout()
    path = _out(f'sample_path_{setting.name}{tag}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_efficiency_bars(summaries: list[dict], tag: str = '') -> Path:
    """Efficiency against the exact optimum, both settings on one axis."""
    # only strategies that appear in every summary, so the bars line up
    names = [name for name in (row['strategy'] for row in summaries[0]['rows'])
             if all(any(r['strategy'] == name for r in summary['rows'])
                    for summary in summaries[1:])]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(summaries))
    x = np.arange(len(names))
    for i, summary in enumerate(summaries):
        lookup = {r['strategy']: r['efficiency'] for r in summary['rows']}
        values = [lookup[name] for name in names]
        ax.bar(x + i * width, values, width, label=summary['setting'])
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.axhline(0.0, color='k', lw=0.8, ls='-')
    ax.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('efficiency = (J - J_naive) / (J_opt - J_naive)')
    ax.set_title('Fraction of the available improvement each strategy captures')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    path = _out(f'efficiency{tag}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
