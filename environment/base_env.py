import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

# from tensorforce import Environment

from utils import day2date

TRADE_UNIT = 100

class BaseEnv():
    """
    """
    def __init__(
            self, 
            initial_value=0, 
            max_episode_timesteps=1000,
            data_dir='./data', 
            raw_dir='./raw',
            log=1, 
            experiment_name='', 
            **kwargs
            ):
        super().__init__()
        self.name = ''
        self.initial_value = initial_value
        self.__max_episode_timesteps__=max_episode_timesteps
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.log = log
        self.exp_name = experiment_name

        # Trade accounting, reset at the start of every episode (see _reset_counters).
        self._reset_counters()

    '''
        You need to overload these functions
    '''   

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError
    
    def action2order(self):
        raise NotImplementedError
    
    def get_state_at_t(self, t):
        raise NotImplementedError  

    def get_reward(self, trade_price, trade_volume):
        # Define reward function here
        reward = self.value - self.value_
        self.value_ = self.value
        return reward

    '''
        Load data
    '''    

    def load_orderbook(self, code, day):
        ask = pd.read_csv(self.data_dir + f'/{code}/{day}/ask.csv')
        bid = pd.read_csv(self.data_dir + f'/{code}/{day}/bid.csv').drop(['timestamp'], axis = 1)

        self.orderbook = pd.concat([ask, bid], axis=1)
        self.orderbook.timestamp = pd.to_datetime(self.orderbook.timestamp)
        self.orderbook = self.orderbook[(f'{self.day} 09:30:00'<self.orderbook.timestamp)&(self.orderbook.timestamp<f'{self.day} 14:57:00')]
        self.orderbook = self.orderbook.set_index('timestamp')
        self.orderbook_length = len(self.orderbook)
        print('load lob done!', code, day)

    def load_orderqueue(self, code, day):
        pass

    def load_price(self, code, day):
        self.price = pd.read_csv(self.data_dir + f'/{code}/{day}/price.csv')
        self.price.timestamp = pd.to_datetime(self.price.timestamp)
        self.price = self.price.set_index('timestamp')
        self.price = self.price.loc[self.orderbook.index]

    def load_msg(self, code, day):
        self.msg = pd.read_csv(self.data_dir + f'/{code}/{day}/msg.csv')
        self.msg.timestamp = pd.to_datetime(self.msg.timestamp)
        self.msg = self.msg.set_index('timestamp')
        self.msg = self.msg.loc[self.orderbook.index]
    
    def load_order(self, code, day):
        order_columns = pd.read_csv(f'{self.raw_dir}/GTA_SZL2_ORDER.csv')
        self.order = pd.read_csv(f'{self.raw_dir}/SZL2_ORDER_{code}_{day[:6]}.csv', names=list(order_columns), low_memory=False)
        self.order.TradingTime = pd.to_datetime(self.order.TradingTime)
        self.order = self.order[self.order.TradingDate==int(day)]
        self.order = self.order[(f'{self.day} 09:30:00'<self.order.TradingTime)&(self.order.TradingTime<f'{self.day} 14:57:00')]

    def load_trade(self, code, day):
        trade_columns = pd.read_csv(f'{self.raw_dir}/GTA_SZL2_TRADE.csv')
        self.trade = pd.read_csv(f'{self.raw_dir}/SZL2_TRADE_{code}_{day[:6]}.csv', names=list(trade_columns))
        self.trade.TradingTime = pd.to_datetime(self.trade.TradingTime)
        self.trade = self.trade[self.trade.TradingDate==int(day)]
        self.trade = self.trade[self.trade.TradeType=="F"]
        self.trade = self.trade[(f'{self.day} 09:30:00'<self.trade.TradingTime)&(self.trade.TradingTime<f'{self.day} 14:57:00')]
        
        self.is_trade = pd.DataFrame(index=self.orderbook.index,columns=['is_trade'])
        self.is_trade['is_trade'] = 0
        # pandas >= 2 rejects a set as a .loc indexer, pass a list instead.
        self.is_trade.loc[list(set(self.trade.TradingTime))] = 1

    '''
        Common function
    '''

    def _reset_counters(self):
        self.cash = self.value_ = self.value = self.initial_value
        self.holding_pnl_total = 0
        self.trading_pnl_total = 0
        self.inventory = 0
        self.inventory_ = 0
        # Total traded notional (CNY), the denominator of the paper's Profit Ratio.
        self.volume = 0
        # Total traded shares, useful for fill-rate style diagnostics.
        self.traded_units = 0
        self.episode_reward = 0
        self.mid_price_ = None
        self.action_his = []
        self.reward_dampened_pnl = 0
        self.reward_trading_pnl = 0
        self.reward_inventory_punishment = 0
        self.reward_spread_punishment = 0

    def _prepare_episode(self, episode_start, episode_end):
        """Slice an episode and build the trade-tick iterator.

        Returns the first state, or None when the window cannot host a single decision
        (fewer than two trade ticks, or a window not longer than the state window T). The
        original implementation raised StopIteration inside reset_seq for such windows.
        """
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.episode_state = self.orderbook.iloc[self.episode_start:self.episode_end]
        self.episode_length = len(self.episode_state)

        episode_is_trade = self.is_trade.iloc[self.episode_start:self.episode_end]
        has_trade_index = np.where(episode_is_trade == 1)[0]
        # A decision at step t reads order book rows [t - T, t) for the LOB state and prices
        # at t - latency for the quotes, so the first usable t is T + latency - 1.
        # With the default latency of 1 this is exactly the original `> self.T`.
        min_index = self.T + self.latency - 1
        has_trade_index = has_trade_index[has_trade_index > min_index]

        if self.episode_length <= self.T:
            if self.log >= 1:
                print(f'Skip env {self.name} {self.code}, {self.day}: episode of '
                      f'{self.episode_length} rows is not longer than T={self.T}')
            return None
        if len(has_trade_index) < 2:
            if self.log >= 1:
                print(f'Skip env {self.name} {self.code}, {self.day}: only '
                      f'{len(has_trade_index)} usable trade tick(s) in the episode')
            return None

        self.index_iterator = iter(has_trade_index)
        self._reset_counters()

        # log for trade
        self.logger = self.price.iloc[self.episode_start:self.episode_end].copy()
        columns=['ask_price', 'bid_price', 'trade_price', 'trade_volume', 'value', 'volume', 'cash', 'inventory']
        for column in columns:
            self.logger[column] = np.nan

        self.i = next(self.index_iterator)
        self.i_ = next(self.index_iterator)
        return self.get_state_at_t(self.i-self.latency)

    def reset_seq(self, timesteps_per_episode=None, episode_idx=None):
        self.episode_idx = episode_idx
        if timesteps_per_episode == None:
            episode_start, episode_end = 0, len(self.orderbook)
        else:
            episode_start = timesteps_per_episode * episode_idx
            episode_end = min(episode_start + timesteps_per_episode, len(self.orderbook))

        state = self._prepare_episode(episode_start, episode_end)
        if state is None:
            return None

        if self.log >= 1:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state
    
    def reset_random(self, timesteps_per_episode=2000):
        max_start = max(1, len(self.orderbook) - timesteps_per_episode)
        episode_start = np.random.randint(0, max_start)
        episode_end = min(episode_start + timesteps_per_episode, len(self.orderbook))

        state = self._prepare_episode(episode_start, episode_end)
        if state is None:
            return None

        if self.log:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state

    def execute(self, actions):
        self.action_his.append(actions)
        # t
        self.mid_price, self.ask1_price, self.bid1_price, self.lob_spread = self.get_price_info(self.i)
        if self.mid_price_ == None:
            self.mid_price_ = self.mid_price

        orders = self.action2order(actions)
        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            orders['ask_price']=0
        elif self.inventory > 10*TRADE_UNIT:
            orders['bid_price']=0

        trade_price, trade_volume = self.match(orders)

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        self.logger.iloc[self.i, -8:] = [orders['ask_price'], orders['bid_price'], trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        # if trade_volume:
        #     print(self.i, 'ask1:', self.ask1_price, 'bid1:', self.bid1_price, 'buy' if trade_volume>0 else 'sell', 'at', trade_price)
        if self.log >= 1:
            self.pbar.update(self.i_ - self.i)
        
        self.i = self.i_
        # Termination conditions
        terminal = False
        try:
            self.i_ = next(self.index_iterator)
        except:
            terminal = True

        reward = self.get_reward(trade_price, trade_volume)
        self.mid_price_ = self.mid_price

        # close position
        if terminal:
            trade_price, trade_volume = self.close_position()
            reward += self.get_reward(trade_price, trade_volume)

        self.episode_reward += reward

        # log for result
        if terminal:
            self.post_experiment()

        state = self.get_state_at_t(self.i-self.latency)
        
        return state, terminal, reward

    def match(self, actions):
        """Fill the agent's quotes against the recorded trade tape.

        Timing convention (unified with `action2order` and `get_price_info`): the quotes
        were placed on prices observed at `t_1 = self.i - self.latency`, and they are filled
        against every trade that arrives in the interval `(index[t_1], index[self.i]]`. With
        the default latency of 1 and one trade per book row this reduces to the original
        behaviour of looking only at the trades stamped at `index[self.i]`.

        Queue model: the agent is assumed to rest at the back of the queue, so a limit order
        at the touch only fills with probability traded_volume / (traded_volume + depth).
        """
        trade_volume = 0
        trade_price = 0
        ask_price, ask_volume, bid_price, bid_volume = actions.values()

        t_1 = self.i - self.latency
        # trade tape over the interval (t_1, i]
        window_start = self.episode_state.index[t_1]
        window_end = self.episode_state.index[self.i]
        now_t = self.trade[(self.trade.TradingTime > window_start)
                           & (self.trade.TradingTime <= window_end)]
        now_trading_price_max = now_t.TradePrice.max()
        now_trading_price_max_v = now_t[now_t.TradePrice==now_trading_price_max].TradeVolume.sum()
        now_trading_price_min = now_t.TradePrice.min()
        now_trading_price_min_v = now_t[now_t.TradePrice==now_trading_price_min].TradeVolume.sum()

        # reference quotes at t - latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(t_1)

        # sell order
        if ask_price and ask_volume:
            if ask_price <= t_1_b1_price:
                # market order
                trade_price, trade_volume = t_1_b1_price, ask_volume
                # print("market order sell at", trade_price)
            else:
                # limit order
                if now_trading_price_max > ask_price:
                    # all deal
                    trade_price, trade_volume = ask_price, ask_volume
                    # print("limit order sell at", trade_price)

                # we assume that our quotes rest at the back of the queue 
                elif now_trading_price_max == ask_price:
                    # deal probability: traded volume/all volume in this level
                    lob_depth = self.episode_state.iloc[self.i].ask1_volume
                    transac_prob = now_trading_price_max_v/(now_trading_price_max_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = ask_price, ask_volume

        # buy order
        if bid_price and bid_volume:
            if bid_price >= t_1_a1_price:
                # market order
                trade_price, trade_volume = t_1_a1_price, bid_volume
                # print("market order buy at", trade_price)
            else:
                if now_trading_price_min < bid_price:
                    trade_price, trade_volume = bid_price, bid_volume
                    # print("limit order buy at", trade_price)

                # we assume that our quotes rest at the back of the queue
                elif now_trading_price_min == bid_price:
                    lob_depth = self.episode_state.iloc[self.i].bid1_volume
                    transac_prob = now_trading_price_min_v/(now_trading_price_min_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = bid_price, bid_volume

        return trade_price, trade_volume
    
    def close_position(self):
        # reference quotes at t - latency, same convention as action2order and match
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        # Market order. Volume accounting is handled by update_agent.
        if self.inventory < 0:
            # Buy
            trade_price, trade_volume = t_1_a1_price, -self.inventory
        elif self.inventory > 0:
            # Sell
            trade_price, trade_volume = t_1_b1_price, -self.inventory
        else:
            trade_price, trade_volume = 0, 0

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        self.logger.iloc[self.i, -6:] = [trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        return trade_price, trade_volume
    
    def update_agent(self, trade_price, trade_volume):
        """Apply a fill to cash, inventory and the PnL decomposition.

        For a fill of `trade_volume` at `trade_price` with mark price `mid` the value change
        splits exactly into

            Delta value = trade_volume * (mid - trade_price)          -> trading PnL (TP)
                        + inventory_before * (mid - mid_previous)     -> holding PnL (HP)

        so `pnl == holding_pnl_total + trading_pnl_total` up to floating point, which
        tests/test_smoke.py asserts. The first fill of an episode has inventory_before == 0,
        so the holding term is zero there by construction.
        """
        self.inventory_ = self.inventory
        self.inventory += trade_volume
        self.cash -= trade_volume*trade_price

        mid_previous = self.mid_price if self.mid_price_ is None else self.mid_price_
        self.trading_pnl_total += trade_volume * (self.mid_price - trade_price)
        self.holding_pnl_total += self.inventory_ * (self.mid_price - mid_previous)

        self.value = self.get_value(self.mid_price)

        # Total traded notional (CNY). The original counted only the buy side.
        self.volume += abs(trade_volume*trade_price)
        self.traded_units += abs(trade_volume)

    def get_price_info(self, i):
        price = self.price[self.price.index==self.episode_state.index[i]]

        bid1_price = price.bid1_price.item()
        ask1_price = price.ask1_price.item()
        bid1_price, ask1_price = round(bid1_price,2), round(ask1_price,2)
        mid_price = (bid1_price+ask1_price)/2
        spread = ask1_price - bid1_price

        return mid_price, ask1_price, bid1_price, spread

    def get_value(self, price):
        return self.cash + self.inventory*price

    '''
        For evaluation and save trading log
    '''

    def post_experiment(self, save=False):
        logger_wo_exit_market = self.logger[(self.logger.ask_price != 0) & (self.logger.bid_price != 0)]
        self.episode_avg_spread = (logger_wo_exit_market.ask_price - logger_wo_exit_market.bid_price).mean()
        self.episode_avg_position = self.logger.inventory.mean()
        self.episode_avg_abs_position = self.logger.inventory.abs().mean()
        self.pnl = self.value - self.initial_value
        # Profit Ratio is PnL / total trading volume, paper section IV-C4. The original used
        # total value rather than PnL in the numerator, which is not a ratio at all. Both
        # ratios are left undefined (nan, not inf) when the denominator is degenerate;
        # pandas sums skip nan, so the aggregated tables are unaffected.
        self.episode_profit_ratio = (self.pnl/self.volume
                                     if self.volume else float('nan'))
        # ND-PnL = PnL / average spread, paper section IV-C4.
        self.nd_pnl = (self.pnl/self.episode_avg_spread
                       if self.episode_avg_spread else float('nan'))
        # PnL-MAP = PnL / mean absolute position, paper section IV-C4.
        self.pnl_map = (self.pnl/self.episode_avg_abs_position
                        if self.episode_avg_abs_position else float('nan'))

        if self.log >= 1:
            print(
                "PnL:", self.pnl, 
                "Holding PnL", self.holding_pnl_total,
                "Trading PnL", self.trading_pnl_total,
                "ND-PnL:", self.nd_pnl,
                "PnL-MAP:", self.pnl_map,
                "Trading volume:", self.volume, 
                "Profit ratio:", self.episode_profit_ratio, 
                "Averaged position:",self.episode_avg_position, 
                "Averaged Abs position:",self.episode_avg_abs_position, 
                "Averaged spread:", self.episode_avg_spread,
                "Episodic reward:", self.episode_reward
                )
            self.pbar.close()

        if self.log >= 2:
            trade_log = self.logger[(self.logger.trade_volume > 0)|(self.logger.trade_volume < 0)]
            for i in range(len(trade_log)):
                item = trade_log.iloc[i]
                if item.trade_volume > 0:
                    print(item.name, 'BUY at', item.trade_price, 'inventory', item.inventory, 'value', item.value)
                elif item.trade_volume < 0:
                    print(item.name, 'SELL at', item.trade_price, 'inventory', item.inventory, 'value', item.value)

        if save:
            now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            log_file = f"./log/{self.exp_name}_{self.code}_{self.day}_{now_time}.csv"
            self.logger.to_csv(log_file)
            print("Trading log saved to", log_file)

    def get_final_result(self):
        return dict(
            pnl=self.pnl, 
            nd_pnl=self.nd_pnl, 
            pnl_map=self.pnl_map,
            profit_ratio=self.episode_profit_ratio, 
            avg_position=self.episode_avg_position, 
            avg_abs_position=self.episode_avg_abs_position, 
            avg_spread=self.episode_avg_spread,
            volume=self.volume, 
            traded_units=self.traded_units,
            holding_pnl=self.holding_pnl_total,
            trading_pnl=self.trading_pnl_total,
            episode_reward=self.episode_reward
        )