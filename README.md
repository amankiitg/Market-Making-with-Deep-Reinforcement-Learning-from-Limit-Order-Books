# Market Making with Deep Reinforcement Learning from Limit Order Books

Demonstration code for the paper "Market Making with Deep Reinforcement Learning from Limit
Order Books" (IJCNN 2023, arXiv:2305.15821). The upstream repository is
[imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books](https://github.com/imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books).

An RL market-making agent quotes a bid and an ask on a Shenzhen level-2 order book. The state
is a window of order book levels, a market feature block (realised volatility, RSI, order
strength indices) and an agent block (normalised inventory and time in episode). The policy
is an Attn-LOB encoder (convolution and inception blocks plus multi-head attention) that can be
pretrained as a mid-price trend classifier, followed by a dense head. Two agents are available
through TensorForce: PPO on a continuous action, and dueling DQN on a discrete action set.

## Read this first: what is NOT in this repository

The two inputs the original code needs were never published, and the code cannot run without
them:

1. **The data.** `data/{code}/{day}/{ask,bid,price,msg}.csv` and `raw/SZL2_{ORDER,TRADE}_*.csv`
   are absent. The paper (section IV-A) uses Shenzhen Stock Exchange level-2 orders and trades
   for November 2019 and does not say the data is public. The `raw/` file names (`GTA_SZL2_*`)
   indicate the CSMAR (Guo Tai An) Shenzhen level-2 dump, which is a licensed product, so it
   cannot be redistributed. Two upstream issues requesting the data are still open.
2. **The pretrained encoder.** `main.py` loads `./ckpt/pretrain_model_{code}/weights`. The
   checkpoint was not published and neither was the script that trains it. This fork adds
   `pretrain.py`, but you still need data to run it.

You can work without real data by generating a synthetic dataset that satisfies the schema,
which is enough to verify the pipeline end to end, not to reproduce paper numbers.

```bash
python -m data.adapter --synthetic --code 000001 --day 20191101 --rows 4000
```

## Install

### macOS (Apple Silicon)

The original `conda_setup.yaml` is a linux-64 / CUDA export and cannot be recreated on arm64.
`tensorforce==0.6.5` likewise declares `tensorflow==2.6.0`, `numpy==1.19.5` and `h5py~=3.1.0`,
none of which resolve on this platform. Use the verified arm64 combination instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
pip install --no-deps tensorforce==0.6.5
pip install pytest==9.1.1
python -c "import tensorflow, tensorforce; print(tensorflow.__version__, tensorforce.__version__)"
# 2.11.0 0.6.5
```

`compat.py` installs a shim for the two Keras optimizer internals that TensorForce calls and
TensorFlow 2.11 removed. It is a no-op on TensorFlow 2.10 and older, and it is imported
automatically by `agent/tensorforce_agent.py` and `main.py`.

### Linux x86_64 (faithful stack)

TensorFlow 2.6 has no arm64 wheel and its x86_64 wheels need AVX, so use a real x86_64 machine
rather than Docker Desktop on Apple Silicon:

```bash
docker build --platform linux/amd64 -t mm-lob:tf26 .
docker run --rm -it --platform linux/amd64 -v "$PWD:/workspace" mm-lob:tf26 bash
```

## Run

```bash
# 1. synthetic data (or drop real data into ./data and ./raw, see data/adapter.py)
python -m data.adapter --synthetic --code 000001 --day 20191101 --rows 4000

# 2. pretrain the Attn-LOB encoder -> ./ckpt/pretrain_model_000001/weights
python pretrain.py --code 000001 --days "['20191101']" --epochs 20

# 3. train and test (writes ./results/{run_id}/ and, with --save, the agent checkpoint)
python main.py --code 000001 \
    --train_days "['20191101']" --test_days "['20191101']" \
    --num_step_per_episode 400 --n_train_loop 1 --save True

# 3b. discrete action, dueling DQN. exploration defaults to 0 in TensorForce, so set it.
python main.py --env_type discrete --agent_type dueling_dqn --exploration 0.3 \
    --train_days "['20191101']" --test_days "['20191101']" \
    --num_step_per_episode 400 --n_train_loop 1

# skip the pretrained encoder (random initialisation), useful before step 2 exists
python main.py --wo_pretrain True --train_days "['20191101']" --test_days "['20191101']"

# tests
pytest
```

Every `TrainConfig` field is a command line flag; run `python main.py --help`. List values
must be given as a Python list literal, for example `--train_days "['20191101', '20191104']"`, because that is how pyrallis decodes list fields.

Outputs land in `./results/{run_id}/`: `config.json`, `train_episodes.csv`,
`test_episodes.csv` and `test_daily.csv` (one row per test day with PnL, ND-PnL, PnL-MAP,
Profit Ratio, position statistics, volume and the holding/trading PnL split).

## Configuration worth knowing

| Flag | Default | Meaning |
| --- | --- | --- |
| `reward_mode` | `composite` | `composite` is the paper reward, Eq. (12) to (16). `pnl` reproduces the shipped code, which used plain PnL plus an undocumented spread penalty. |
| `wo_dampened_pnl`, `wo_matched_pnl`, `wo_inv_punish` | `False` | Remove the dampened, matched and inventory punishment terms from the composite reward. They had no effect in the shipped code because the composite expression was commented out. |
| `train_deterministic` | `False` | Sample actions while training. The shipped loop used `Agent.act`'s default `deterministic=True`, which makes TensorForce skip exploration entirely. Set to `True` to reproduce that behaviour. |
| `test_deterministic` | `True` | Act greedily when testing. |
| `exploration` | `0.0` | TensorForce exploration for the discrete agent, e.g. a float or `dict(type='linear', unit='updates', num_steps=2000, initial_value=1.0, final_value=0.01)`. |
| `discrete_num_actions` | `8` | Paper section III-C1 defines 8 actions; action 7 flattens the position. The shipped code declared 5, which made three of them unreachable. |
| `max_bias`, `max_spread` | `0.05`, `0.1` | Continuous action scaling, paper section IV-C2. |
| `inventory_penalty`, `asymmetry_eta` | `0.01`, `0.5` | Reward coefficients zeta and eta, paper section IV-C2. |
| `latency` | `1` | Decision delay in order book rows, see the docstring of `EnvContinuous.action2order` for the exact timing convention. |

## Data schema

`data/adapter.py` documents the exact schema and offers three entry points:
`generate_synthetic`, `validate_dataset` and a documented (not implemented) `convert_tardis`
skeleton for a crypto level-2 plus trade-tape source. Points to watch when plugging in your own
data:

* The fill model needs a **trade tape**; a book-snapshot-only dataset (for example FI-2010)
  cannot be used.
* `msg.csv` is required whenever the market state is enabled. Snapshot deltas cannot separate
  new limit orders from cancellations, so order strength indices are approximate for any
  source that lacks a true order-by-order feed.
* Trades must be stamped at order book timestamps, otherwise `load_trade` raises.
* The pretraining path filters to `10:00:00 < time < 14:30:00`, while the environment accepts
  `09:30:00` to `14:57:00`. A dataset must cover the mid-session window to be usable for
  pretraining.
* Tick size, lot size, fees and rebates are not modelled. See `convert_tardis` for the full
  mismatch list.

## Tests

```bash
pytest                                  # all tests
pytest tests/test_wiring.py             # state wiring only
```

`tests/test_wiring.py` is the important one. It checks that every declared state reaches the
policy network exactly once, that perturbing any state changes the output, and, end to end
through TensorForce with sentinel values, that the states are fed in canonical order.
`tests/test_smoke.py` covers the episode loop, the PnL decomposition identity, both reward
modes, the ablation flags, the discrete action space, the unusable-window guard and the
pretraining data pipeline.

## Changes relative to the upstream repository

The upstream code as published could not start. Fixes, with the reason:

| File | Change | Why |
| --- | --- | --- |
| `main.py` | `save: bool = False,` to `save: bool = False` | The trailing comma made the default a truthy tuple `(False,)`, so `save_agent` ran unconditionally. |
| `main.py` | `keras_model_dir` is now a config field | It was never defined, so any run that reached `save_agent` or `--load` raised `NameError`. |
| `main.py` | `agent.save(..., filename='agent')` | It passed the agent object as the filename: `TypeError: join() argument must be str`. |
| `main.py` | `train_days`, `test_days`, `num_step_per_episode`, `n_train_loop`, `max_episode_timesteps` moved into `TrainConfig` | They were module globals defined only under `if __name__ == '__main__'`, so the module was not importable and `max_episode_timesteps` was hardcoded to 1000 while episodes were 2000 steps. |
| `main.py` | writes `./results/{run_id}/` | The run produced no output at all: `daily_test_results` was computed and dropped. |
| `environment/base_env.py` | `list(set(...))` | `df.loc[set(...)]` raises on pandas 2, so the environment could not be constructed. |
| `environment/base_env.py` | `reset_seq` returns `None` for unusable windows, and `_prepare_episode` shares the logic with `reset_random` | The old code raised `StopIteration` for quiet windows, and short windows produced a wrongly shaped LOB state. |
| `environment/base_env.py` | reward is configurable, `pnl` splits into holding and trading parts, `volume` counts both sides | Plain PnL meant the paper's reward ablations were inert; holdings PnL was never populated; volume counted buys only. |
| `environment/base_env.py` | `nd_pnl`, `pnl_map` and `profit_ratio` are guarded and now match paper section IV-C4 | They divided by zero on flat episodes and `profit_ratio` used total value rather than PnL. |
| `environment/base_env.py` | one `t - latency` reference in `action2order`, `match` and `close_position` | `match` compared against `t-1` while the action used `t-latency`. |
| `network/network.py` | `dense_input.append(market_state)` in the market-state branch | It appended `agent_state` twice, so the market state was never consumed and the agent state was silently dropped. |
| `network/network.py` | one state order from `state_spec.STATE_KEYS` | The environment emitted lob, market, agent while the network expected lob, agent, market. |
| `network/network.py` | FC-LOB Dense layers chained | All three read the flattened input, so the 1024 and 256 layers were outside the graph and the ablation was a single linear layer. |
| `environment/env_discrete.py` | `num_values` is configurable, default 8 | It was hardcoded to 5 while `action2order` implemented actions 0 to 7, so the flatten action was unreachable. |
| `environment/env_continuous.py` | action bounds are `[0, 1]` | The spec declared `[-1, 1]` while the implementation assumed `[0, 1]`, which allowed negative spreads (crossed quotes). |
| `state_spec.py`, `utils.py` | canonical LOB channel order | The channel order was whatever the CSV happened to contain, and the unused `utils.reorder` documented yet another order. Both the RL and pretraining paths now use `state_spec.LOB_COLUMNS`. |
| `compat.py` | Keras optimizer shim | TensorForce 0.6.5 calls `_create_all_weights` and `_create_hypers`, removed in TensorFlow 2.11. |
| `pretrain.py` | new | The pretraining script was missing from the repository. |
| `data/adapter.py` | new | Schema documentation, a validation helper and a synthetic generator. |
| `agent/tensorforce_agent.py` | `exploration`, `eager_mode` pass-through | TensorForce's default exploration is 0, and exploration is skipped when `deterministic` is true. |

## Known gaps

* Hyperparameters the paper does not state (learning rate, batch size, discount, memory,
  pretraining loss and epochs) are marked `# ASSUMED` in `pretrain.py` or carried over from the
  shipped code. The paper also does not print its exact train/test calendar dates, and the
  shipped lists (8 train, 13 test days) do not match the paper's described 10/11 day split.
* Volume channels are normalised by the window maximum on the RL path and by the dataset
  maximum on the pretraining path, so pretrained weights see a different volume scale. See the
  note in `pretrain.py`.
* `holding_pnl + trading_pnl == pnl` is asserted by the tests, but `holding_pnl` is not
  reported by the upstream code and has not been compared against any published figure.
* No baselines (Avellaneda-Stoikov, fixed quoting, inventory RL) are included, so agent results
  cannot be compared against the paper's tables from this repository alone.
