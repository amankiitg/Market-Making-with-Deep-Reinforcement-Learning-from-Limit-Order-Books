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
    """Corporate bond dealer answering client RFQs, logistic hit ratio.

    Units: the price is quoted per 100 of face value, and one lot is 1mm of face value, so a
    price move of 1 cent per 100 of face is 100 USD on one lot, i.e.
    usd_per_price_unit = 1_000_000 / 100 = 10_000 USD per price unit per lot.
    """
    sigma = 0.25   # price units per day, see the rationale below
    # gamma is chosen so that one lot held for the whole day costs about twice the spread
    # earned on one fill: 0.5 * gamma * sigma^2 * T = 2.25 * (spread per fill), where the
    # spread per fill at the optimal depth is about 0.0139 price units. That puts the
    # inventory penalty at the same order of magnitude as the revenue it protects, which is
    # what makes inventory control matter in this setting.
    gamma = 1.0
    phi = 0.5 * gamma * sigma ** 2
    return Setting(
        name='credit',
        market='one investment grade corporate bond, RFQ dealing, logistic hit ratio',
        s0=100.0,
        sigma=sigma,
        T=1.0,
        n_steps=200,
        lam=50.0,
        Q=10,
        gamma=gamma,
        phi=phi,
        a=phi * 1.0,
        fill_kind='logistic',
        k=None,
        alpha=0.0,
        beta=100.0,
        d_max=0.06,
        usd_per_price_unit=10_000.0,
        pnl_label='USD',
        price_label='price per 100 of face value',
    )


def parameter_table(setting: Setting) -> list[dict]:
    """Rows for the README: parameter, value, unit, source or ASSUMED with a rationale."""
    common = [
        dict(parameter='s0', value=setting.s0, unit=setting.price_label,
             source='equity: Avellaneda and Stoikov (2008); credit: par bond, ASSUMED'),
        dict(parameter='sigma', value=setting.sigma,
             unit='price units per unit time',
             source=('equity: Avellaneda and Stoikov (2008). credit: ASSUMED, duration about 5 '
                     'years times a daily yield move of about 5 bp gives 0.25 per 100 of face')),
        dict(parameter='T', value=setting.T, unit='time',
             source='equity: as in the paper. credit: one trading day'),
        dict(parameter='dt', value=setting.dt, unit='time',
             source='equity: as in the paper. credit: same discretisation'),
        dict(parameter='n_steps', value=setting.n_steps, unit='steps',
             source='T / dt'),
        dict(parameter='lam', value=setting.lam, unit='arrivals per unit time',
             source=('equity: Avellaneda and Stoikov (2008) A = 140. credit: ASSUMED, 50 '
                     'requests per side per day for one liquid investment grade bond')),
        dict(parameter='Q', value=setting.Q, unit='lots',
             source=('chosen so the bound rarely binds; tests and the README report the '
                     'fraction of time at the bound')),
        dict(parameter='gamma', value=setting.gamma, unit='risk aversion',
             source=('equity: Avellaneda and Stoikov (2008) numerical section. credit: chosen '
                     'so that 0.5 gamma sigma^2 T, the cost of holding one lot for the whole '
                     'day, is about 2.25 times the spread earned on one fill; see the class '
                     'docstring for credit_setting')),
        dict(parameter='phi', value=setting.phi, unit='price per unit time per lot^2',
             source='derived: 0.5 * gamma * sigma^2'),
        dict(parameter='a', value=setting.a, unit='price per lot^2',
             source='derived: phi * T, so ending with q lots costs the same as holding q lots '
                    'for the whole horizon'),
        dict(parameter='usd_per_price_unit', value=setting.usd_per_price_unit, unit='USD',
             source=('reporting only, it never enters the dynamics. equity: one unit of '
                     'inventory is one share, reported per 100 share lot. credit: one lot is '
                     'one million of face, quoted per 100 of face, so one price unit is 10000 '
                     'USD per lot')),
        dict(parameter='d_max', value=setting.d_max, unit='price units',
             source='RL action bound. Quoting inside this bound is worth within 0.001 percent of '
                    'the unbounded optimum (see tests/test_benchmark.py and the README table), '
                    'and it puts the optimal depth near a third of the action range so that '
                    'exploration covers the region that matters'),
    ]
    if setting.fill_kind == 'exponential':
        common.append(dict(parameter='k', value=setting.k, unit='1 / price',
                           source='Avellaneda and Stoikov (2008)'))
    else:
        common.append(dict(parameter='alpha', value=setting.alpha, unit='dimensionless',
                           source='ASSUMED, gives a hit ratio of 0.5 at a zero half-spread'))
        common.append(dict(parameter='beta', value=setting.beta, unit='1 / price unit',
                           source=('ASSUMED. beta sets the optimal half-spread through the '
                                   'first order condition beta (1 - f) d = 1, so beta = 100 '
                                   'puts the optimal half-spread near 1.4 cents per 100 of '
                                   'face, i.e. 1.4 basis points, with a hit ratio near 0.20, '
                                   'the order of magnitude reported for liquid investment '
                                   'grade dealer-to-client RFQ activity')))
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


def zero_inventory_depths(setting: Setting, h_result) -> tuple[float, float]:
    """Optimal depths at t = 0 for q = 0, in price units. Useful as a sanity print."""
    q_index = int(np.where(h_result.q == 0)[0][0])
    return float(h_result.d_ask[0, q_index]), float(h_result.d_bid[0, q_index])
