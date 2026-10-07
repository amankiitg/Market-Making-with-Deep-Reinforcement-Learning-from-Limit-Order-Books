"""Parameter tables for the two settings.

Every parameter records its unit and where it comes from. Values marked ASSUMED could not be
sourced and were chosen with the stated rationale. See the README section for the rendered
table.

Shared conventions
------------------
* q is an integer inventory in LOTS and Q is the symmetric inventory limit.
* Depths d_b, d_a are measured from the mid price in "price units" (see usd_per_price_unit).
* phi is the running inventory penalty (per unit time) and a the terminal penalty, both in
  whatever unit the price is quoted in.
* phi and a are derived with the same dimensionless recipe in both settings:
      phi = 0.5 * gamma * sigma^2      (gamma is the risk-aversion parameter)
      a   = phi * T
  For the equity setting this reproduces the value suggested in the task brief,
  phi = 0.5 * 0.1 * 2^2 = 0.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Setting:
    name: str
    market: str
    s0: float
    sigma: float
    T: float
    n_steps: int
    lam: float
    Q: int
    gamma: float
    phi: float
    a: float
    fill_kind: str
    k: float | None
    alpha: float | None
    beta: float | None
    d_max: float
    # Multiply a raw objective value by this to report it in a familiar unit. The model's unit
    # of inventory is one share for equity and one million of face for credit; equity PnL is
    # reported per 100 share lot, the usual lot size, which is presentation only.
    usd_per_price_unit: float
    pnl_label: str
    price_label: str

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def fill_model(self):
        """The fill intensity model f for this setting (see benchmark.hjb)."""
        from benchmark.hjb import make_fill

        return make_fill(self.fill_kind, k=self.k, alpha=self.alpha, beta=self.beta)


def equity_setting() -> Setting:
    """Avellaneda-Stoikov (2008) calibration, exponential fill intensity.

    s0 = 100, sigma = 2, T = 1, dt = 0.005, A (lam) = 140, k = 1.5 and gamma = 0.1 are the
    values used in the paper's numerical section. phi and a follow the shared recipe.
    """
    sigma = 2.0
    gamma = 0.1
    phi = 0.5 * gamma * sigma ** 2
    return Setting(
        name='equity',
        market='single equity, continuous mid price, exponential fill intensity',
        s0=100.0,
        sigma=sigma,
        T=1.0,
        n_steps=200,
        lam=140.0,
        Q=20,
        gamma=gamma,
        phi=phi,
        a=phi * 1.0,
        fill_kind='exponential',
        k=1.5,
        alpha=None,
        beta=None,
        d_max=2.0,
        usd_per_price_unit=100.0,
        pnl_label='USD',
        price_label='price (one share)',
    )


def credit_setting() -> Setting:
    """One corporate bond dealer answering client requests for quote, logistic hit ratio.

    Units and market calibration
    ---------------------------
    One lot is 1mm of face value and the price is quoted per 100 of face, so one price unit
    (one point) on one lot is 1_000_000 / 100 = 10_000 USD and the reporting scale below is
    10_000. Modified duration D = 7 gives a DV01 of 700 USD per lot per basis point, and one
    basis point of spread is therefore 700 / 10_000 = 0.07 price points.

    Worked conversion of the calibration target:
        inputs:  D = 7, half-spread target = 2.5 bp
        output:  price half-spread = 7 * 0.00025 * 100 = 0.175 points = 1_750 USD per lot
    The 0.175 point figure is what the solved optimum quotes at (t = 0, q = 0), and
    tests/test_benchmark.py asserts it.

    Sources of each number are in parameter_table below. sigma comes from a spread volatility
    of 4 bp per day, which is ASSUMED, and lam (10 requests per side per day) is ASSUMED. The
    hit ratio parameters (alpha, beta) are not assumed: they are solved numerically so that
    the dynamic program's optimal half-spread at (t = 0, q = 0) is 0.175 points and the hit
    ratio at that depth is 0.30, both matching the stated targets.
    """
    sigma = 0.28        # 4 bp per day times 0.07 points per bp
    # The risk aversion rule is unchanged: the cost of holding one lot for a full day,
    # 0.5 * gamma * sigma^2 * T, is about 2.25 times the half-spread earned on one fill at the
    # optimal depth (0.175 points), so gamma = 2 * 2.25 * 0.175 / sigma^2.
    phi = 2.25 * 0.175
    gamma = 2.0 * phi / sigma ** 2
    return Setting(
        name='credit',
        market='one investment grade corporate bond, RFQ dealing, logistic hit ratio',
        s0=100.0,
        sigma=sigma,
        T=1.0,
        n_steps=200,
        lam=10.0,
        Q=4,
        gamma=gamma,
        phi=phi,
        a=phi * 1.0,
        fill_kind='logistic',
        k=None,
        alpha=-1.627238,
        beta=14.140205,
        d_max=0.7,
        usd_per_price_unit=10_000.0,
        pnl_label='USD',
        price_label='price per 100 of face value',
    )


def parameter_table(setting: Setting) -> list[dict]:
    """Rows for the README: parameter, value, unit, source or ASSUMED with a rationale."""
    common = [
        dict(parameter='s0', value=setting.s0, unit=setting.price_label,
             source='equity: Avellaneda and Stoikov (2008). credit: par bond at 100, ASSUMED'),
        dict(parameter='sigma', value=setting.sigma,
             unit='price units per unit time',
             source=('equity: Avellaneda and Stoikov (2008). credit: ASSUMED, a spread '
                     'volatility of 4 bp per day times 0.07 points per bp gives 0.28 points')),
        dict(parameter='T', value=setting.T, unit='time',
             source='equity: as in the paper. credit: one trading day'),
        dict(parameter='dt', value=setting.dt, unit='time',
             source='equity: as in the paper. credit: same discretisation'),
        dict(parameter='n_steps', value=setting.n_steps, unit='steps',
             source='T / dt'),
        dict(parameter='lam', value=setting.lam, unit='arrivals per unit time',
             source=('equity: Avellaneda and Stoikov (2008) A = 140. credit: ASSUMED, 10 '
                     'requests per side per day for one liquid investment grade bond')),
        dict(parameter='Q', value=setting.Q, unit='lots',
             source=('chosen so the bound never binds: the optimal policy reaches |q| = 2 at '
                     'most and spends no measurable time at the bound')),
        dict(parameter='gamma', value=round(setting.gamma, 4), unit='risk aversion',
             source=('equity: Avellaneda and Stoikov (2008) numerical section. credit: derived '
                     'from 0.5 gamma sigma^2 T = 2.25 * 0.175, i.e. holding one lot for a full '
                     'day costs 2.25 times the half-spread earned on one fill')),
        dict(parameter='phi', value=setting.phi, unit='price per unit time per lot^2',
             source='derived: 0.5 * gamma * sigma^2'),
        dict(parameter='a', value=setting.a, unit='price per lot^2',
             source='derived: phi * T, so ending with q lots costs the same as holding q lots '
                    'for the whole horizon'),
        dict(parameter='a in bp of spread', value=round(setting.a / 0.07, 2),
             unit='bp',
             source='same number expressed per basis point of spread: a / 0.07'),
        dict(parameter='usd_per_price_unit', value=setting.usd_per_price_unit, unit='USD',
             source=('reporting only, it never enters the dynamics. equity: one unit of '
                     'inventory is one share, reported per 100 share lot. credit: one lot is '
                     'one million of face, quoted per 100 of face, so one price unit is 10000 '
                     'USD per lot')),
        dict(parameter='d_max', value=setting.d_max, unit='price units',
             source='RL action bound, chosen inside the band that keeps the optimal depth at '
                    '(t=0, q=0) between 20 and 50 percent of the range and a bound cost under '
                    '0.1 percent of the optimum; both are measured, see the README'),
    ]
    if setting.fill_kind == 'exponential':
        common.append(dict(parameter='k', value=setting.k, unit='1 / price',
                           source='Avellaneda and Stoikov (2008)'))
    else:
        common.append(dict(parameter='alpha', value=setting.alpha, unit='dimensionless',
                           source=('SOLVED, not assumed: calibrated jointly with beta so that '
                                   'the dynamic program quotes a 2.5 bp half-spread at '
                                   '(t=0, q=0) and the hit ratio there is 30 percent. Hit '
                                   'ratio levels for investment grade RFQ are UNVERIFIED, see '
                                   'the README')))
        common.append(dict(parameter='beta', value=setting.beta, unit='1 / price unit',
                           source=('SOLVED, not assumed: the slope follows from the first '
                                   'order condition beta (1 - f) d = 1 once alpha is fixed by '
                                   'the 30 percent hit ratio target')))
    return common


# The RL action is a pair of numbers in [-1, 1]^2 in the order (ask, bid). A depth uses
# d_max * sigmoid(LOGIT_SCALE * action), so the whole action box maps onto depths from
# 0.018 * d_max to 0.982 * d_max, well outside the optimal depths in both settings, and the
# mapping is smooth everywhere so the policy gradient stays informative.
LOGIT_SCALE = 4.0


def depths_from_action(setting: 'Setting', action) -> tuple[np.ndarray, np.ndarray]:
    """Map an action in [-1, 1]^2, ordered (ask, bid), to depths in price units."""
    action = np.asarray(action, dtype=float)
    squashed = setting.d_max / (1.0 + np.exp(-LOGIT_SCALE * action))
    return squashed[..., 0], squashed[..., 1]


def analytic_reward_scale(setting: 'Setting') -> float:
    """A fixed reward scale that uses only model inputs, never the solved optimum.

    The scale is 1 / (lam * T * d_foc), where d_foc is the depth that maximises f(d) d for the
    fill model, i.e. the first-order condition of the fill model at a zero price of inventory
    risk. For the exponential model d_foc = 1/k; for the logistic model it solves
    beta (1 - f(d)) d = 1. Two properties make this a legitimate scale rather than a leak: it
    depends only on lam, T and the fill model, and it is the same for both settings by
    construction. Multiplying a reward by a positive constant leaves every optimal policy
    unchanged, so it affects learning only through the size of the critic's targets, which is
    the whole point: without it the critic regresses returns of order fifty.
    """
    d_foc = float(setting.fill_model().optimal_depth(0.0))
    return 1.0 / (setting.lam * setting.T * d_foc)


def zero_inventory_depths(setting: Setting, h_result) -> tuple[float, float]:
    """Optimal depths at t = 0 for q = 0, in price units. Useful as a sanity print."""
    q_index = int(np.where(h_result.q == 0)[0][0])
    return float(h_result.d_ask[0, q_index]), float(h_result.d_bid[0, q_index])
