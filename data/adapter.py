"""Dataset adapter: the one place that knows the on-disk schema the environments read.

Layout expected by `environment/base_env.py`
--------------------------------------------
```
data/{code}/{day}/ask.csv     timestamp, ask{i}_price, ask{i}_volume       for i = 1..10
data/{code}/{day}/bid.csv     timestamp, bid{i}_price, bid{i}_volume       for i = 1..10
data/{code}/{day}/price.csv   timestamp, ask1_price, bid1_price, midprice, spread,
                              last_price, trading_price, trading_volume
data/{code}/{day}/msg.csv     timestamp, market_buy_volume, market_buy_n,
                              market_sell_volume, market_sell_n,
                              limit_buy_volume, limit_buy_n,
                              limit_sell_volume, limit_sell_n,
                              withdraw_buy_volume, withdraw_buy_n,
                              withdraw_sell_volume, withdraw_sell_n
raw/GTA_SZL2_TRADE.csv        header-only file listing the trade column names
raw/SZL2_TRADE_{code}_{YYYYMM}.csv   headerless trade rows, same column order
raw/GTA_SZL2_ORDER.csv        header-only file listing the order column names
raw/SZL2_ORDER_{code}_{YYYYMM}.csv   headerless order rows (optional, see below)
```
`{day}` is `YYYYMMDD`; the directory uses that form while the in-file `timestamp` uses
`YYYY-MM-DD HH:MM:SS`. `code` is a 6 digit instrument code.

What the environments actually consume
--------------------------------------
* `ask.csv`/`bid.csv` supply the 40 LOB channels. Their column *names* must be exactly the
  ones above; the on-disk column *order* is normalised internally to
  `state_spec.LOB_COLUMNS`, so any order is fine.
* `price.csv` supplies the mid price, the volume mark and the best quotes used for quotes,
  fills and marking to market.
* `msg.csv` is only read when the market state is enabled (`wo_market_state=False`). It
  feeds the order strength indices in `utils.getOrderStrengthIndex`.
* `raw/SZL2_TRADE_*` is read by `load_trade` to decide *when* an agent order can fill; the
  fill model needs a trade tape, so a dataset without trades cannot be used.
* `raw/SZL2_ORDER_*` is only used by `BaseEnv.load_order`, which no environment calls. The
  adapter therefore does not require order files, see `convert_tardis` for the caveat.

Timestamps
----------
Every row of `ask`/`bid`/`price`/`msg` must share the same timestamp index for a given day
(`load_price` and `load_msg` reindex onto the order book). Trade timestamps should coincide
with order book timestamps where possible: `load_trade` marks trade ticks by timestamp and
raises if a trade timestamp is absent from the order book index.

Command line
------------
Generate a synthetic dataset for smoke tests and for `pretrain.py`:

    python -m data.adapter --synthetic --code 000001 --day 20191101 --rows 4000
"""

import argparse
import os

import numpy as np
import pandas as pd

ASK_BID_COLUMNS = (
    ['timestamp']
    + [f'ask{i}_{field}' for i in range(1, 11) for field in ('price', 'volume')]
)
BID_COLUMNS = (
    ['timestamp']
    + [f'bid{i}_{field}' for i in range(1, 11) for field in ('price', 'volume')]
)
PRICE_COLUMNS = [
    'timestamp', 'ask1_price', 'bid1_price', 'midprice', 'spread', 'last_price',
    'trading_price', 'trading_volume'
]
MSG_COLUMNS = [
    'timestamp',
    'market_buy_volume', 'market_buy_n', 'market_sell_volume', 'market_sell_n',
    'limit_buy_volume', 'limit_buy_n', 'limit_sell_volume', 'limit_sell_n',
    'withdraw_buy_volume', 'withdraw_buy_n', 'withdraw_sell_volume', 'withdraw_sell_n',
]
# Column names from the exchange export dump. Only the first 14 are needed by the fill
# model, the rest are carried through so the file matches the original vendor layout.
TRADE_COLUMNS = [
    'Symbol', 'TradingDate', 'TradingTime', 'BuyOrderID', 'SellOrderID', 'TradePrice',
    'TradeVolume', 'TradeType', 'UNIX', 'Market', 'SecurityID', 'SymbolSource', 'SetID',
    'RecID'
]
ORDER_COLUMNS = [
    'Symbol', 'TradingDate', 'TradingTime', 'SetID', 'RecID', 'OrderPrice', 'OrderVolume',
    'OrderType', 'UNIX', 'Market', 'SymbolSource', 'OrderCode', 'SecurityID'
]

# Trade types that do not represent an executed fill. The reference implementation filters
# on TradeType == "F" and the paper does not document the code space, so treat "F" as the
# only fill type and map everything else away.
FILL_TRADE_TYPE = 'F'


def _check_columns(frame, expected, name):
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f'{name} is missing required column(s): {missing[:8]}')
    extra = [column for column in frame.columns if column not in expected]
    if extra:
        raise ValueError(
            f'{name} has unexpected column(s): {extra[:8]}. Drop or rename them; the '
            f'environments index these frames positionally through the schema in '
            f'data/adapter.py.'
        )


def validate_dataset(data_root, raw_root, code, day):
    """Check that on-disk files for one day satisfy the schema. Raises ValueError."""
    day_dir = os.path.join(data_root, code, day)
    frames = {}
    for name, columns in (('ask.csv', ASK_BID_COLUMNS), ('bid.csv', BID_COLUMNS),
                          ('price.csv', PRICE_COLUMNS), ('msg.csv', MSG_COLUMNS)):
        path = os.path.join(day_dir, name)
        if not os.path.exists(path):
            raise ValueError(f'missing required file {path}')
        frame = pd.read_csv(path)
        _check_columns(frame, columns, path)
        frames[name] = frame

    lengths = {name: len(frame) for name, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f'ask/bid/price/msg must have equal length, got {lengths}')
    if len(set(tuple(frame.timestamp) for frame in frames.values())) != 1:
        raise ValueError('ask/bid/price/msg must share the same timestamps')

    trade_header = os.path.join(raw_root, 'GTA_SZL2_TRADE.csv')
    trade_file = os.path.join(raw_root, f'SZL2_TRADE_{code}_{day[:6]}.csv')
    for path in (trade_header, trade_file):
        if not os.path.exists(path):
            raise ValueError(f'missing required file {path}')
    header = list(pd.read_csv(trade_header))
    if header != TRADE_COLUMNS:
        raise ValueError(f'{trade_header} must list exactly {TRADE_COLUMNS}, got {header}')
    trades = pd.read_csv(trade_file, names=header)
    if len(trades) == 0:
        raise ValueError(f'{trade_file} has no trades; the fill model needs a trade tape')
    return frames


def write_day(data_root, raw_root, code, day, ask, bid, price, msg, trades):
    """Write one day of data in the schema above.

    `trades` must have the columns listed in TRADE_COLUMNS except that only Symbol,
    TradingDate, TradingTime, TradePrice, TradeVolume, TradeType are used by the
    environment; the remaining columns are written as given (they can be zero).
    """
    _check_columns(ask, ASK_BID_COLUMNS, 'ask')
    _check_columns(bid, BID_COLUMNS, 'bid')
    _check_columns(price, PRICE_COLUMNS, 'price')
    _check_columns(msg, MSG_COLUMNS, 'msg')
    missing = [column for column in TRADE_COLUMNS if column not in trades.columns]
    if missing:
        raise ValueError(f'trades is missing required column(s): {missing}')

    day_dir = os.path.join(data_root, code, day)
    os.makedirs(day_dir, exist_ok=True)
    ask.to_csv(os.path.join(day_dir, 'ask.csv'), index=False)
    bid.to_csv(os.path.join(day_dir, 'bid.csv'), index=False)
    price.to_csv(os.path.join(day_dir, 'price.csv'), index=False)
    msg.to_csv(os.path.join(day_dir, 'msg.csv'), index=False)

    os.makedirs(raw_root, exist_ok=True)
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(
        os.path.join(raw_root, 'GTA_SZL2_TRADE.csv'), index=False)
    # headerless on purpose: base_env.load_trade passes names= explicitly
    trades[TRADE_COLUMNS].to_csv(
        os.path.join(raw_root, f'SZL2_TRADE_{code}_{day[:6]}.csv'), index=False, header=False)
    return day_dir


def generate_synthetic(data_root='./data', raw_root='./raw', code='000001', day='20191101',
                       rows=4000, tick_size=0.01, lot_size=100, trade_stride=5, seed=0,
                       start_price=10.0, start_time='10:00:00'):
    """Write a synthetic dataset: a random walk mid price with 10 book levels per side.

    The values are meaningless, the point is to exercise the schema end to end. The default
    start time is 10:00:00 because utils.process_data, which pretrain.py uses, keeps only
    rows between 10:00:00 and 14:30:00, while BaseEnv.load_orderbook accepts 09:30:00 to
    14:57:00. Trades are
    emitted every `trade_stride` book rows, always at a book timestamp, which is what
    `load_trade` requires. Prices land on the `tick_size` grid and volumes are multiples of
    `lot_size`, like the Shenzhen market the paper uses.
    """
    rng = np.random.default_rng(seed)
    date = f'{day[:4]}-{day[4:6]}-{day[6:]}'
    timestamps = pd.date_range(f'{date} {start_time}', periods=rows, freq='s')

    mid = start_price + np.cumsum(rng.normal(0, tick_size / 5.0, rows))
    mid = np.round(mid / tick_size) * tick_size

    ask = {'timestamp': timestamps}
    bid = {'timestamp': timestamps}
    for i in range(1, 11):
        ask[f'ask{i}_price'] = np.round(mid + i * tick_size, 2)
        ask[f'ask{i}_volume'] = rng.integers(1, 6, rows) * lot_size
        bid[f'bid{i}_price'] = np.round(mid - i * tick_size, 2)
        bid[f'bid{i}_volume'] = rng.integers(1, 6, rows) * lot_size

    price = pd.DataFrame({
        'timestamp': timestamps,
        'ask1_price': np.round(mid + tick_size, 2),
        'bid1_price': np.round(mid - tick_size, 2),
        'midprice': np.round(mid, 2),
        'spread': 2 * tick_size,
        'last_price': np.round(mid, 2),
        'trading_price': np.round(mid, 2),
        'trading_volume': rng.integers(0, 5, rows) * lot_size,
    })

    msg = {'timestamp': timestamps}
    for column in MSG_COLUMNS[1:]:
        msg[column] = rng.integers(0, 20, rows)

    # Trades print at the touch: a buyer-initiated fill takes ask1, a seller-initiated fill
    # takes bid1. This matters, the fill model in BaseEnv.match only fills a resting quote
    # when a trade prints at or through its price. Trade sizes are larger than the top-of-book
    # depth so that a back-of-queue quote has a realistic fill probability.
    #
    # Some trades walk one level deeper (ask2 / bid2), which is what happens in the real book
    # when an aggressive order is larger than the best level. Without them, quotes wider than
    # one tick never fill and every episode is a zero-PnL no-op.
    trade_index = np.arange(0, rows, trade_stride)
    buyer_initiated = rng.random(len(trade_index)) < 0.5
    level_offset = rng.integers(1, 3, len(trade_index)) * tick_size
    trade_price = np.where(
        buyer_initiated, mid[trade_index] + level_offset, mid[trade_index] - level_offset)
    trades = pd.DataFrame({
        'Symbol': code,
        'TradingDate': int(day),
        'TradingTime': timestamps[trade_index].strftime('%Y-%m-%d %H:%M:%S'),
        'BuyOrderID': rng.integers(1, 10 ** 6, len(trade_index)),
        'SellOrderID': rng.integers(1, 10 ** 6, len(trade_index)),
        'TradePrice': np.round(trade_price, 2),
        'TradeVolume': rng.integers(5, 20, len(trade_index)) * lot_size,
        'TradeType': FILL_TRADE_TYPE,
        'UNIX': 0, 'Market': 'SZ', 'SecurityID': code, 'SymbolSource': 'SZL2',
        'SetID': 0, 'RecID': 0,
    })

    day_dir = write_day(
        data_root, raw_root, code, day,
        pd.DataFrame(ask), pd.DataFrame(bid), price, pd.DataFrame(msg), trades)
    print(f'synthetic dataset written to {day_dir} '
          f'({rows} book rows, {len(trades)} trades)')
    return day_dir


def convert_tardis(book_snapshots_path, trades_path, data_root='./data', raw_root='./raw',
                   code='BTCUSD', day=None, tick_size=None, lot_size=None,
                   book_levels=10, snapshot_freq='1s'):
    """SKELETON: convert a crypto L2 snapshot file plus a trade tape to the schema.

    Expected inputs (Tardis style: one CSV or CSV.GZ per day per channel):
      * book snapshots: one row per snapshot with a timestamp column and per level columns
        for price and amount on both sides, 25 levels is the usual depth;
      * trades: one row per trade with a timestamp, a side, a price and an amount.

    Column naming differs between vendors and between Tardis releases, so this function does
    not guess: pass the frame through `rename` to the names used below before calling, or
    extend the mapping. Nothing here is verified against a live Tardis download.

    Schema mismatches this converter CANNOT fix, all of which change results:
      * Order queue file. There is no order-by-order file in a snapshot plus trade feed, so
        the queue position model in `BaseEnv.match` (which assumes the agent rests at the
        back of the queue) cannot be validated against real queue data. The reference data
        came with `SZL2_ORDER_*` files, which the environments do not currently use either.
      * Order strength index. `msg.csv` needs market / limit / withdraw volumes and counts.
        Book deltas only give the net change per level, so new limit orders and cancellations
        are indistinguishable. The numbers written below are therefore approximate and the
        market state will not match the paper's features unless a true order feed is used.
      * `price.csv` fields last_price / trading_price / trading_volume have no exact
        counterpart in a book snapshot channel; they are filled from the trade tape.
      * Tick size and lot size differ per instrument and venue. Quotes and fill levels are
        rounded to the tick grid in `utils.price_legal_check`, so passing the wrong tick size
        silently changes every quote. Pass the real values.
      * Timestamps. The paper's data is event time on a single venue; crypto feeds are
        usually microsecond UTC. Downsample to `snapshot_freq` and make sure the trade
        timestamps coincide with book timestamps, otherwise `load_trade` raises.
      * Fees, rebates and market impact are not modelled anywhere in this repository. A
        crypto conversion without adding them will overstate PnL.
      * Session boundaries. Shenzhen data is one 09:30 to 14:57 session; crypto trades 24/7.
        One file per day is required because `load_orderbook` filters within a single day.
    """
    if tick_size is None or lot_size is None:
        raise ValueError('tick_size and lot_size are required: they change every quote and '
                         'fill and cannot be inferred from the file')
    if day is None:
        raise ValueError('day (YYYYMMDD) is required, the loader keys files by day')

    snapshots = pd.read_csv(book_snapshots_path)
    trades_raw = pd.read_csv(trades_path)

    raise NotImplementedError(
        'convert_tardis is a documented skeleton, not a working converter. Implement the '
        'steps below for your vendor, then call data.adapter.validate_dataset and '
        'data.adapter.write_day:\n'
        '  1. rename the snapshot columns to timestamp, ask{i}_price, ask{i}_volume, '
        'bid{i}_price, bid{i}_volume and keep book_levels levels;\n'
        '  2. drop na / crossed rows and round prices to tick_size, amounts to lot_size;\n'
        '  3. build price.csv from the snapshot (midprice = (ask1+bid1)/2) and fill '
        'last_price / trading_price / trading_volume from the trade tape;\n'
        '  4. build msg.csv as documented in the mismatch list above, and record it as an '
        'approximation;\n'
        '  5. map trades to TRADE_COLUMNS with TradeType = FILL_TRADE_TYPE, TradingDate = '
        'int(day) and TradingTime aligned to the snapshot grid;\n'
        '  6. write_day(data_root, raw_root, code, day, ask, bid, price, msg, trades).'
    )


def main():
    parser = argparse.ArgumentParser(description='Dataset adapter, see module docstring.')
    parser.add_argument('--synthetic', action='store_true',
                        help='write a synthetic dataset')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--raw-root', default='./raw')
    parser.add_argument('--code', default='000001')
    parser.add_argument('--day', default='20191101')
    parser.add_argument('--rows', type=int, default=4000)
    parser.add_argument('--trade-stride', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate', action='store_true',
                        help='validate an existing dataset instead of writing one')
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic(
            data_root=args.data_root, raw_root=args.raw_root, code=args.code, day=args.day,
            rows=args.rows, trade_stride=args.trade_stride, seed=args.seed)
    if args.validate:
        validate_dataset(args.data_root, args.raw_root, args.code, args.day)
        print(f'dataset for {args.code} {args.day} is schema-valid')
    if not args.synthetic and not args.validate:
        parser.error('nothing to do: pass --synthetic and/or --validate')


if __name__ == '__main__':
    main()
