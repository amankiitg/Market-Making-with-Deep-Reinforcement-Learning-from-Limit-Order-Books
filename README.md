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

## Market-making benchmark: RL vs the known optimum (equity and credit RFQ)

`benchmark/` is a self-contained, under twenty minute, laptop CPU benchmark that answers one
question honestly: can an ordinary RL agent recover a market-making policy whose optimum is
known exactly, and how much of the available edge does it capture? Nothing here needs the
Shenzhen data, and nothing here depends on the reproduction below.

It has two settings. Both are a single dealer with an inventory limit who posts a bid and an ask
and is filled by a Poisson arrival process whose intensity decays with the distance of the quote
from the mid:

| | equity | credit |
| --- | --- | --- |
| story | one cash equity, arithmetic mid price, exponential fill intensity | one investment grade corporate bond, request-for-quote dealing, logistic hit ratio |
| fill intensity fitting the classic model | `f(d) = exp(-k d)`, `k = 1.5` | `f(d) = 1 / (1 + exp(beta d))`, `beta = 100` |
| known optimum | Cartea-Jaimungal closed form, and the discrete optimum below | no closed form, discrete optimum only |
| parameters | Avellaneda-Stoikov (2008) numerical section | assumed, see the parameter table |

### What is "the known optimum" here, exactly

Three independent answers are computed, and the report shows all three so that a mistake in any
one of them would be visible.

1. **The HJB solution.** With inventory `q`, time `t`, running penalty `phi q^2` and terminal
   penalty `a q^2`, writing the value function as `x + q s + h(t, q)` gives

   ```
   d/dt h(t,q) = phi q^2 - lam H(h(t,q) - h(t,q-1)) - lam H(h(t,q) - h(t,q+1))
   h(T,q) = -a q^2,   H(p) = sup over d >= 0 of f(d) (d - p)
   ```

   solved with an implicit Runge-Kutta integrator and an analytic Jacobian
   (`benchmark/hjb.py`). For the exponential setting there is also the exact matrix-exponential
   closed form, `h = (1/k) log w`, `w = expm(M (T-t)) z`, which the solver reproduces to 1e-9 at
   every grid point.

2. **The exact discrete optimum.** The simulator moves inventory at most one lot up and one lot
   down per step, so the expected objective satisfies a finite backward recursion on the grid of
   `(step, inventory)` that can be solved exactly (`benchmark/dp.py`). Its value at an empty book
   is the largest expected PnL any policy can achieve in the simulator, with no Monte Carlo error.
   This is the number the agent is graded against.

3. **The optimal policy as a table.** Both of the above produce `d_ask` and `d_bid` for every
   cell of the `(time, inventory)` grid, which is what the simulator replays.

Two things fell out of validating these against each other, and both are pinned by tests:

* **The floor at a zero depth matters, so the textbook closed form is not the answer.** `H`
  maximises over `d >= 0`. With these parameters, an inventory near the limit makes the price of
  inventory risk more negative than `-1/k`, so the exact optimum quotes at the mid on the
  reducing side, where the exponential branch of `H` is outside its range. The closed form solves
  the corresponding unfloored system: it agrees with the exact solution to a few hundredths of a
  price unit near a flat book and differs by tens of units near the inventory limit
  (`tests/test_hjb.py::test_floor_at_zero_depth_separates_exact_solution_from_closed_form`).
  The optimum used everywhere below is the floored one.
* **The continuous-time optimum and the discrete optimum are close but not equal.** The exact
  discrete optimum is `J*_discrete`, and the HJB policy achieves `J_HJB`, a fraction of a percent
  below it. The difference is a discretisation effect, mostly because the exact one-step fill
  probability `1 - exp(-lam f dt)` is smaller than the intensity `lam f dt` that the ODE uses.
  The grader is `J*_discrete`; `J_HJB` is reported next to it.

### Evaluation protocol

* The tuned baseline is the best **constant** quote depth, chosen on separate tuning episodes
  and never on the test episodes.
* Every headline number is produced on 2000 held out episodes with **common random numbers**, so
  strategies are compared on the same price shocks and the same fill draws. Differences are then
  estimated far more precisely than the individual means.
* The headline score is `efficiency = (J_strategy - J_naive) / (J_optimum - J_naive)`, the
  fraction of the gap between a tuned constant quote and the exact optimum that a strategy closes.
* The RL entry is an ensemble of five seeds: its reported per-episode value is the average over
  the five policies, which is exactly the value of the mixed policy that picks a seed uniformly.
* The agent must not beat the optimum beyond sampling error. `run_all.py` prints a paired test of
  every strategy against the optimal policy, so a violation would be visible rather than averaged
  away.

### Results

| strategy | mean PnL (USD per 100 share lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **exact optimum** (dynamic program) | 5832.6 | +-32.3 | 7.90 | 0.993 | n/a | 0.89 | 80.2 |
| HJB solver policy | 5802.7 | +-31.3 | 8.13 | 0.974 | -29.9 +- 11.2 | 0.94 | 88.4 |
| textbook closed-form policy | 5801.8 | +-31.3 | 8.12 | 0.973 | -30.8 +- 11.2 | 0.96 | 88.4 |
| Avellaneda-Stoikov heuristic | 5494.2 | +-29.2 | 8.25 | 0.775 | -338.4 +- 21.1 | 2.28 | 85.3 |
| **RL**, PPO, mean of 5 seeds | 5588.7 | +-29.0 | 8.44 | 0.836 | -243.9 +- 12.6 | 1.81 | 61.3 |
| RL, worst of the 5 seeds | 5189.3 | +-40.9 | 5.56 | 0.578 | -643.3 +- 35.6 | 0.82 | 79.9 |
| tuned constant quote (baseline) | 4294.9 | +-103.5 | 1.82 | 0.000 | -1537.7 +- 97.2 | 5.89 | 66.2 |
| deep fixed quote | 2290.8 | +-43.0 | 2.34 | -1.295 | -3541.8 +- 44.9 | 2.89 | 13.5 |

| strategy | mean PnL (USD per 1mm face lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **exact optimum** (dynamic program) | 1877.1 | +-108.8 | 0.76 | 0.996 | n/a | 0.23 | 16.8 |
| HJB solver policy | 1878.1 | +-109.5 | 0.75 | 0.996 | +1.1 +- 14.9 | 0.25 | 17.0 |
| **RL**, PPO, mean of 5 seeds | 1536.9 | +-76.5 | 0.88 | 0.812 | -340.1 +- 61.2 | 0.52 | 17.4 |
| RL, worst of the 5 seeds | 1274.3 | +-113.3 | 0.49 | 0.670 | -602.8 +- 111.6 | 0.45 | 10.2 |
| tuned constant quote (baseline) | 35.2 | +-58.8 | 0.03 | 0.000 | -1841.9 +- 118.8 | 0.44 | 0.6 |
| deep fixed quote | 42.1 | +-38.6 | 0.05 | 0.004 | -1834.9 +- 113.1 | 0.21 | 0.2 |

| setting | reward used in training | RL efficiency | RL mean PnL |
| --- | --- | --- | --- |
| equity | variance reduced (default) | 0.836 | 5588.7 |
| equity | fully realised PnL | 0.828 | 5576.6 |
| credit | variance reduced (default) | 0.812 | 1536.9 |
| credit | fully realised PnL | 0.397 | 770.0 |

Figures are written to `figures/`: optimal depths against inventory for both solvers,
end of day inventory distributions, learning curves against the optimum, a sample episode, and
the efficiency bars.

### Parameters, with units and provenance

| parameter | equity | credit | unit | source |
| --- | --- | --- | --- | --- |
| `s0` | 100.0 | 100.0 | price (one share) | equity: Avellaneda and Stoikov (2008); credit: par bond, ASSUMED |
| `sigma` | 2.0 | 0.25 | price units per unit time | equity: Avellaneda and Stoikov (2008). credit: ASSUMED, duration about 5 years times a daily yield move of about 5 bp gives 0.25 per 100 of face |
| `T` | 1.0 | 1.0 | time | equity: as in the paper. credit: one trading day |
| `dt` | 0.005 | 0.005 | time | equity: as in the paper. credit: same discretisation |
| `n_steps` | 200 | 200 | steps | T / dt |
| `lam` | 140.0 | 50.0 | arrivals per unit time | equity: Avellaneda and Stoikov (2008) A = 140. credit: ASSUMED, 50 requests per side per day for one liquid investment grade bond |
| `Q` | 20 | 10 | lots | chosen so the bound rarely binds; tests and the README report the fraction of time at the bound |
| `gamma` | 0.1 | 1.0 | risk aversion | equity: Avellaneda and Stoikov (2008) numerical section. credit: chosen so that 0.5 gamma sigma^2 T, the cost of holding one lot for the whole day, is about 2.25 times the spread earned on one fill; see the class docstring for credit_setting |
| `phi` | 0.2 | 0.03125 | price per unit time per lot^2 | derived: 0.5 * gamma * sigma^2 |
| `a` | 0.2 | 0.03125 | price per lot^2 | derived: phi * T, so ending with q lots costs the same as holding q lots for the whole horizon |
| `usd_per_price_unit` | 100.0 | 10000.0 | USD | reporting only, it never enters the dynamics. equity: one unit of inventory is one share, reported per 100 share lot. credit: one lot is one million of face, quoted per 100 of face, so one price unit is 10000 USD per lot |
| `d_max` | 2.0 | 0.06 | price units | RL action bound. Quoting inside this bound is worth within 0.001 percent of the unbounded optimum (see tests/test_benchmark.py and the README table), and it puts the optimal depth near a third of the action range so that exploration covers the region that matters |
| `k` | 1.5 |  | 1 / price | Avellaneda and Stoikov (2008) |
| `alpha` |  | 0.0 | dimensionless | ASSUMED, gives a hit ratio of 0.5 at a zero half-spread |
| `beta` |  | 100.0 | 1 / price unit | ASSUMED. beta sets the optimal half-spread through the first order condition beta (1 - f) d = 1, so beta = 100 puts the optimal half-spread near 1.4 cents per 100 of face, i.e. 1.4 basis points, with a hit ratio near 0.20, the order of magnitude reported for liquid investment grade dealer-to-client RFQ activity |

Anything marked ASSUMED is a choice, not a measurement. The two that carry the most weight are
the credit hit-ratio slope `beta` and the risk aversion `gamma`: `beta` sets the optimal
half-spread through the first order condition `beta (1 - f(d)) d = 1`, and `gamma` sets how much
inventory risk costs. `gamma` for credit is chosen so that holding one lot for a whole day costs
about 2.25 times the spread earned on a single fill, which is what makes inventory control
worth modelling in that setting; with a smaller value the optimal policy becomes nearly
inventory-blind and a constant quote almost matches it, which would make the benchmark
uninformative.

### Design choices, and the honest list

* **The RL action space is `[0, d_max]^2`, and `d_max` is tight on purpose.** The unconstrained
  optimum wants very deep quotes at the inventory limit near the close (up to 8.5 price units for
  equity). Solving the same problem with a bound four times wider changes the optimum by less
  than 0.001 percent for equity and 0.02 percent for credit, so the bound is not the binding
  constraint on the answer, but it does make exploration cover the region that matters. Both
  numbers are asserted in `tests/test_benchmark.py::test_the_action_bound_is_almost_free`. The
  agent is graded against the exact optimum **of its own bounded problem**.
* **The marking noise is removed from the training reward, not from the reported PnL.** The
  realised step reward contains `q_{k+1} (S_{k+1} - S_k)`, a martingale increment independent of
  the fills. Its expectation is zero under every policy and it is larger than the spread earned
  per step, so it adds no information about the policy while dominating the learning signal. The
  default environment replaces `S_{k+1}` with its conditional mean, which makes the expected step
  reward exactly the expression the dynamic program optimises. Reported PnL for every strategy
  comes from the full simulator, price shocks included. `run_all.py --ablation` trains on the
  fully realised reward for comparison; the second table below shows what it costs. The variance reduction is worth little in the equity setting (efficiency 0.828 against 0.836) and a great deal in the credit setting (0.397 against 0.812), because the credit spread earned per fill is small next to the daily price risk of one lot.
* **The mid price is a martingale, so it cannot move the expected objective, only the dispersion.**
  With fills independent of the price path and a policy that does not condition on the price, the
  level of `S` enters the objective linearly and cancels. A test checks that the optimum is
  unchanged when `sigma` is set to zero. The price shocks matter for the reported PnL dispersion
  and for the noise in the learning signal, which is why they are kept.
* **At most one fill per side per step.** The one-step fill probability is the exact probability
  of at least one Poisson event, `1 - exp(-lam f dt)`, which understates the fill count by
  `O((lam f dt)^2)` relative to an event-driven simulation. At the equity parameters that is
  about 11 percent of fills at the open. Inventory, cash and the depth trade-off are unaffected
  in expectation.
* **The observation is `(t / T, q / Q)`.** That is not a shortcut: the optimal policy is a
  function of time and inventory only, in both settings, because the mid price enters the value
  function linearly. The agent therefore has every feature the optimum uses, and no strategy is
  given a feature another does not get.
* **One hyperparameter needed a decision.** The initial exploration width `log_std_init = -1.0`
  instead of the stable-baselines3 default of 0. At the default, sampled actions span the whole
  action box, which for credit means almost every sample is a quote too deep to trade and the
  agent receives almost no signal; on two seeds, 300k steps, credit efficiency was 0.43 and 0.59
  at the default and 0.94 and 0.78 at `-1.0`. Everything else is stock PPO.

### Reproduce

```bash
pip install -r benchmark/requirements.txt
python -m benchmark.run_all               # full run, about 20 minutes on a MacBook CPU
python -m benchmark.run_all --quick       # smoke run, about a minute
pytest benchmark/tests                    # 33 tests, about 2.5 minutes with the slow ones
```

`benchmark/results/summary.json` holds every number that produced the tables above, including the
per-seed objectives, the tuning sweep and the paired tests.


Wall clock for the whole run, including training, evaluation and figures: 18.7 minutes on an Apple Silicon MacBook CPU, single process, no GPU.

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
* No baselines (Avellaneda-Stoikov, fixed quoting, inventory RL) are wired into the paper
  reproduction, so agent results cannot be compared against the paper's tables from this
  repository alone. The separate benchmark described at the top of this file does implement
  those baselines, but on its own two stylised settings rather than on the paper's data.
