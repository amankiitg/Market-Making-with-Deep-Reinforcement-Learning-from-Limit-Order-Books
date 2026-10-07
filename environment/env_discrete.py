import numpy as np
import pandas as pd
import random
import math
import time
from datetime import timedelta

from .env_feature import EnvFeature
from .base_env import TRADE_UNIT

from utils import day2date, lob_norm
from state_spec import active_state_keys, state_specs

class EnvDiscrete(EnvFeature):
    """
    """
    def __init__(
            self, 
            code='000001', 
            day='20191101', 
            data_norm=True, 
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
            # action space
            num_actions=8,
            **kwargs
        ):
        super().__init__(**kwargs)
        print("Environment: EnvDiscrete")
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

        # Paper section III-C1 defines 8 discrete actions, 0..7, where action 7 closes the
        # position with market orders. action2order below implements exactly those 8, so a
        # smaller value silently removes the flatten action from the action space.
        if not 1 <= num_actions <= 8:
            raise ValueError(
                f'num_actions must be between 1 and 8 (implemented actions are 0..7), '
                f'got {num_actions}'
            )
        self.num_actions = num_actions

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
        return dict(
                    type='int',
                    num_values=self.num_actions
                )
    
    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        """Map a discrete action to a bid/ask quote pair (paper section III-C1).

        Actions 0..6 place a symmetric limit pair on the 0.01 yuan tick grid, widening the
        pair as the index grows (level 0 at the touch, level 2 two ticks deep). Action 7
        sends market orders that flatten the current inventory. Volume is TRADE_UNIT (100
        shares) per side.

        Timing convention (unified with `match` and `get_price_info`): at decision step `i`
        the agent may only use information up to `t_1 = self.i - self.latency`; the quotes
        below are placed on `t_1` reference prices and `match` fills them against the trades
        in `(t_1, i]`. See EnvContinuous.action2order for a worked latency example.
        """
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        ask_price, bid_price = 0, 0
        ask_volume, bid_volume = -TRADE_UNIT,TRADE_UNIT

        if actions in range(7):
            # limit order
            if actions == 0:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price
            elif actions == 1:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.01
            elif actions == 2: 
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price
            elif actions == 3: 
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price-0.01
            elif actions == 4: 
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.02
            elif actions == 5: 
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price
            elif actions == 6: 
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price-0.02

        elif actions==7:   
            # market order to clode position
            if self.inventory < 0:
                bid_price, bid_volume = np.inf, -self.inventory
            elif self.inventory > 0:
                ask_price, ask_volume = 0.01, -self.inventory
            else:
                trade_price, trade_volume = 0, 0

        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            ask_price=0
            ask_volume=0
        elif self.inventory > 10*TRADE_UNIT:
            bid_price=0
            bid_volume=0
        
        orders = {
            'ask_price': ask_price,
            'ask_vol': ask_volume,
            'bid_price': bid_price,
            'bid_vol': bid_volume
        }

        return orders

    def get_reward(self, trade_price, trade_volume):
        """Reward for one environment step, see EnvContinuous.get_reward for the equations.

        reward_mode='composite' is the paper's reward, Eq. (12)-(16) in section III-D, with
        the inventory punishment in its L2 form IP = theta * (inventory / TRADE_UNIT) ** 2
        (paper Eq. (15)). The shipped discrete env used an L1 form on the inventory change,
        but that line was commented out, so nothing that was actually active is lost.

        reward_mode='pnl' reproduces the shipped code path: reward = DeltaPnL.
        """
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL, paper Eq. (13)
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        # Matched (trading) PnL, paper Eq. (14)
        matched_pnl = (self.mid_price - trade_price) * trade_volume

        # Inventory punishment, paper Eq. (15)
        inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2

        if self.reward_mode == 'composite':
            reward = (self.r_ma * matched_pnl
                      + self.r_da * dampened_pnl
                      - self.r_ip * inventory_punishment)
        elif self.reward_mode == 'pnl':
            reward = pnl
        else:
            raise ValueError(
                f"reward_mode must be 'composite' or 'pnl', got {self.reward_mode!r}"
            )

        # Reward components, surfaced in get_final_result for PnL attribution.
        self.reward_dampened_pnl = dampened_pnl
        self.reward_trading_pnl = matched_pnl
        self.reward_inventory_punishment = inventory_punishment
        self.reward_spread_punishment = 0

        self.value_ = self.value

        return reward

    def get_state_at_t(self, t):
        self.__state__ = dict()

        # Built in canonical state order, see state_spec.STATE_KEYS.
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
