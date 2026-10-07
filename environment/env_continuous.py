import numpy as np
import pandas as pd
import random
import math
import time

from .env_feature import EnvFeature
from .base_env import TRADE_UNIT

from utils import day2date, lob_norm, price_legal_check
from state_spec import active_state_keys, state_specs

class EnvContinuous(EnvFeature):
    """
    """
    def __init__(
            self, 
            code='000001', 
            day='20191101', 
            latency=1, 
            T=50, 
            # ablation states
            wo_lob_state=False,
            wo_market_state=False,
            wo_agent_state=False,
            # ablation rewards
            wo_dampened_pnl=False,
            wo_matched_pnl=False,
            wo_inv_punish=False,
            # reward definition
            reward_mode='composite',
            inventory_penalty=0.01,
            asymmetry_eta=0.5,
            spread_penalty_lambda=100.0,
            # quote parameterisation (paper section IV-C2: max_bias=0.05, max_spread=0.1)
            max_bias=0.05,
            max_spread=0.1,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.name = "Continuous"
        print("Environment:", self.name)
        self.code = code
        self.day = day2date(day)

        self.latency = latency
        self.T = T

        # ablation
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_agent_state = wo_agent_state
        self.r_da = 0 if wo_dampened_pnl else 1
        self.r_ma = 0 if wo_matched_pnl else 1
        self.r_ip = 0 if wo_inv_punish else 1

        self.reward_mode = reward_mode
        # Inventory punishment factor zeta, paper Eq. (15) and section IV-C2.
        self.theta = inventory_penalty
        # Asymmetry factor eta, paper Eq. (13) and section IV-C2.
        self.eta = asymmetry_eta
        # Multiplier of the non-paper spread penalty that the shipped code added to the
        # plain-PnL reward. Only used when reward_mode == 'pnl'; 0 disables it.
        self.spread_penalty_lambda = spread_penalty_lambda
        self.max_bias = max_bias
        self.max_spread = max_spread

        # Set by action2order, read by get_reward.
        self.reservation = None
        self.spread = 0.0

        self.init_states()

        self.load_orderbook(code=code, day=day)
        self.load_price(code=code, day=day)
        self.load_trade(code=code, day=day)
        self.load_msg(code=code, day=day)

    def init_states(self):
        self.state_keys = active_state_keys(
            wo_lob_state=self.wo_lob_state,
            wo_market_state=self.wo_market_state,
            wo_agent_state=self.wo_agent_state,
        )
        self.__states_space__ = state_specs(self.T, keys=self.state_keys)

    def states(self):
        return self.__states_space__

    def actions(self):
        # Paper section III-C2, Eq. (8)-(10): both action components live in [0, 1].
        # A1 scales the inventory skew by max_bias, A2 scales the half-spread by max_spread.
        # The shipped code declared [-1, 1] while implementing [0, 1] semantics, which would
        # have allowed a negative spread (crossed quotes) for A2 < 0.
        return dict(
                    type='float',
                    shape=(2,),
                    min_value=0.0,
                    max_value=1.0
                )

    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        """Map a continuous action to a bid/ask quote pair.

        Timing convention (unified with `match` and `get_price_info`): at decision step `i`
        the agent may only use information up to `t_1 = self.i - self.latency`. All reference
        prices below come from `t_1`, and `match` fills these quotes against the trades that
        arrive in the interval `(t_1, i]`.

        Worked example for `latency = 1` and the 5th decision step (self.i = 9):
        * decision information: order book and prices at t_1 = 8, i.e. `get_price_info(8)`
        * quotes are placed relative to `mid_price(t_1=8)`
        * `match` compares them with the trade tape over `(index[8], index[9]]`
        * mark to market at the end of the step uses the mid price at `i = 9`
        """
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        # action 1: inventory skew, paper section III-C2 Eq. (8)-(9)
        # actions in [0, 1]
        delta_price = actions[0]*self.max_bias
        # action 2: half spread, paper section III-C2 Eq. (10)
        spread = actions[1]*self.max_spread
        if self.inventory > 0:
            reservation = t_1_mid_price - delta_price
        elif self.inventory < 0:
            reservation = t_1_mid_price + delta_price
        else:
            reservation = t_1_mid_price
        ask_price = reservation + spread/2
        bid_price = reservation - spread/2

        # action 2
        # actions in [-1, 1]
        # delta_price = actions[0]*0.05
        # spread = abs(actions[1])*0.1
        # reservation = t_1_mid_price - delta_price
        # ask_price = reservation + spread/2
        # bid_price = reservation - spread/2

        # action 3
        # actions in [0, 1]
        # ask_price = t_1_a1_price + actions[0]*0.1
        # bid_price = t_1_b1_price - actions[1]*0.1
        # reservation = (ask_price + bid_price)/2
        # spread = ask_price - bid_price

        ask_price, bid_price = price_legal_check(ask_price, bid_price)

        # save for log
        self.reservation = reservation
        self.spread = spread
        
        orders = {
            'ask_price': ask_price,
            'ask_vol': -TRADE_UNIT,
            'bid_price': bid_price,
            'bid_vol': TRADE_UNIT
        }
        return orders

    def get_reward(self, trade_price, trade_volume):
        """Reward for one environment step.

        reward_mode='composite' (default, paper Eq. (12)-(16) in section III-D):

            DeltaPnL  = value_t - value_(t-dt)                                  (Eq. 12)
            DP        = DeltaPnL - max(0, eta * DeltaPnL)                      (Eq. 13)
            TP        = trade_volume * (mid_price - trade_price)                (Eq. 14)
            IP        = theta * (inventory / TRADE_UNIT) ** 2                   (Eq. 15)
            R         = r_ma * TP + r_da * DP - r_ip * IP                       (Eq. 16)

        The r_* multipliers are 1 unless the corresponding wo_* ablation flag is set, which
        is what makes the reward ablations in the paper actually change the reward.

        reward_mode='pnl' reproduces the shipped code path: plain DeltaPnL, plus (continuous
        env only) an undocumented spread penalty applied while flat. The spread penalty does
        not appear in the paper; set spread_penalty_lambda=0 for a clean plain-PnL baseline.
        """
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL, paper Eq. (13)
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        # Matched (trading) PnL, paper Eq. (14)
        matched_pnl = (self.mid_price - trade_price) * trade_volume

        # Inventory punishment, paper Eq. (15). Inventory is measured in TRADE_UNIT lots so
        # that theta=0.01 has the same scale as the paper's zeta.
        inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2

        # Spread penalty, NOT part of the paper's reward. Only used in 'pnl' mode.
        if self.inventory:
            spread_punishment = 0
        else:
            spread_punishment = (self.spread_penalty_lambda * self.spread
                                 if self.spread > 0.02 else 0)

        if self.reward_mode == 'composite':
            reward = (self.r_ma * matched_pnl
                      + self.r_da * dampened_pnl
                      - self.r_ip * inventory_punishment)
        elif self.reward_mode == 'pnl':
            reward = pnl - spread_punishment
        else:
            raise ValueError(
                f"reward_mode must be 'composite' or 'pnl', got {self.reward_mode!r}"
            )

        # Reward components, surfaced in get_final_result for PnL attribution.
        self.reward_dampened_pnl = dampened_pnl
        self.reward_trading_pnl = matched_pnl
        self.reward_inventory_punishment = inventory_punishment
        self.reward_spread_punishment = spread_punishment

        self.value_ = self.value

        return reward

    def get_state_at_t(self, t):
        self.__state__ = dict()

        # Built in canonical state order, see state_spec.STATE_KEYS. TensorForce feeds the
        # Keras model with list(states.values()), so this order must match network.get_model.
        for key in self.state_keys:
            if key == 'lob_state':
                lob = self.episode_state.iloc[t-self.T:t]
                mid_price = (lob.ask1_price + lob.bid1_price)/2
                lob_normed = lob_norm(lob, mid_price)
                self.__state__['lob_state'] = np.expand_dims(np.array(lob_normed), -1)
            elif key == 'market_state':
                self.__state__['market_state'] = self._get_market_state(t) + self._get_order_strength_index(t)
            elif key == 'agent_state':
                self.__state__['agent_state'] = [self.inventory/(10*TRADE_UNIT)]*12 + [t / self.episode_length]*12

        return self.__state__
