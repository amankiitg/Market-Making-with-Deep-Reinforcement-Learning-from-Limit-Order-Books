Archived. The paper's data was never released, so this reproduction cannot be completed. The follow-on benchmark lives in mm-rl-vs-optimum.

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

## Market-making benchmark: RL vs the known optimum (equity and corporate-bond RFQ)

`benchmark/` is a self-contained, under twenty minute, laptop CPU benchmark that answers one
question honestly: can an ordinary RL agent recover a market-making policy whose optimum is
known exactly, and how much of the available edge does it capture? Nothing here needs the
Shenzhen data, and nothing here depends on the reproduction below.

Two settings, both a single dealer with an inventory limit who posts a bid and an ask and is
filled by a Poisson arrival process whose intensity decays with the distance of the quote from
the mid:

| | equity | credit |
| --- | --- | --- |
| story | one cash equity, arithmetic mid price, exponential fill intensity | one corporate bond, request-for-quote dealing with a logistic hit ratio |
| fill intensity | `f(d) = exp(-k d)`, `k = 1.5` | `f(d) = 1 / (1 + exp(alpha + beta d))` |
| known optimum | Cartea-Jaimungal closed form, and the discrete optimum below | discrete optimum only, no closed form |
| parameters | Avellaneda-Stoikov (2008) numerical section | calibrated in spread terms, see below |

### What "the known optimum" means here

Three independent answers are computed, and the report shows all three so that a mistake in any
one of them would be visible.

1. **The HJB solution.** With inventory `q`, time `t`, running penalty `phi q^2` and terminal
   penalty `a q^2`, writing the value function as `x + q s + h(t, q)` gives

   ```
   d/dt h(t,q) = phi q^2 - lam H(h(t,q) - h(t,q-1)) - lam H(h(t,q) - h(t,q+1))
   h(T,q) = -a q^2,   H(p) = sup over d >= 0 of f(d) (d - p)
   ```

   solved with an implicit Runge-Kutta integrator and an analytic Jacobian
   (`benchmark/hjb.py`). For the exponential setting the exact matrix exponential closed form,
   `h = (1/k) log w` with `w = expm(M (T-t)) z`, is available and the solver reproduces it to
   1e-9 at every grid point.

2. **The exact discrete optimum.** The simulator moves inventory at most one lot up and one lot
   down per step, so the expected objective satisfies a finite backward recursion on the grid of
   `(step, inventory)` that is solved exactly (`benchmark/dp.py`). Its value at an empty book is
   the largest expected PnL any policy can achieve in the simulator, with no Monte Carlo error.
   This is the number the agent is graded against.

3. **The optimal policy as a table.** Both of the above produce `d_ask` and `d_bid` for every
   cell of the `(time, inventory)` grid, which is what the simulator replays.

Two properties of this problem fell out of cross-checking the three, and both are pinned by
tests. First, **the floor at a zero depth changes the answer**: `H` maximises over `d >= 0`, and
near the inventory limit the exact optimum quotes at the mid on the reducing side, where the
exponential branch of `H` is outside its range. The textbook closed form solves the
corresponding unfloored system, so it agrees with the exact solution to a few hundredths of a
price unit near a flat book and differs by tens of units near the inventory limit. Second, **the
continuous-time and discrete optima differ slightly**: the exact discrete optimum is what is
graded, and the HJB policy lands a fraction of a percent below it, mostly because the exact
one-step fill probability `1 - exp(-lam f dt)` is smaller than the intensity `lam f dt`.

### The credit setting in dealer terms

One lot is 1mm of face value and the price is quoted per 100 of face, so one price point on one
lot is 10,000 USD. Modified duration `D = 7` gives a DV01 of 700 USD per lot per basis point,
which puts one basis point of spread at 0.07 price points.

Worked example, inputs and outputs labelled:

```
inputs:  D = 7, target half-spread = 2.5 bp
output:  price half-spread = D * 0.00025 * 100 = 7 * 0.00025 * 100 = 0.175 points
         in dollars = 0.175 * 10,000 = 1,750 USD of spread on one 1mm lot
```

The dealer faces about 10 requests per side per day, quotes a two-way price, and is filled on a
side with probability `f(d)` where `d` is the distance of that quote from the mid. Holding one
lot for a whole day costs `0.5 gamma sigma^2 T`, which is set to 2.25 times the 1,750 USD of
spread earned on a single fill, so inventory risk is a first-order concern: the solving policy
keeps the inventory within plus or minus two lots and quotes widely on the side that would
worsen a position.

The hit-ratio parameters are not guessed. They are solved so that the dynamic program quotes
exactly the 2.5 basis point half-spread at `(t = 0, q = 0)` and the hit ratio at that depth is
30 percent, and `tests/test_benchmark.py` re-checks both targets.

### Evaluation protocol

* The tuned baseline is the best **constant** quote depth, chosen on separate tuning episodes
  and never on the test episodes.
* Every headline number is produced on 2000 held out episodes with **common random numbers**, so
  strategies are compared on the same price shocks and the same fill draws.
* The headline score is `efficiency = (J_strategy - J_naive) / (J_optimum - J_naive)`, the
  fraction of the gap between a tuned constant quote and the exact optimum that a strategy closes.
* Five training seeds per setting. Every headline efficiency is the 5-seed mean plus or minus the
  standard deviation across seeds, with the worst and best seeds reported as well. In the credit
  setting one seed in five collapses to a policy that barely trades, which is visible in the
  worst seed column and is a consequence of the sparse fill signal rather than a bug.
* The agent must not beat the optimum beyond sampling error. `run_all.py` prints a paired test of
  every strategy against the optimal policy, so a violation would be visible rather than averaged
  away.
* **The optimum is never visible to the training loop.** No reward scaling, callback, stopping
  rule or hyperparameter uses the optimum, the dynamic program or the HJB solution: those appear
  only in evaluation, reporting and tests, and
  `tests/test_benchmark.py::test_training_cannot_see_the_optimum` enforces that by parsing the
  trainer and the environment. Reward size during training is controlled either by
  stable-baselines3 reward normalization (`VecNormalize` with `norm_obs=False`,
  `norm_reward=True`, statistics saved next to the agent) or by a fixed constant derived from
  the fill model, `1 / (lam * T * d_foc)` where `d_foc` solves the fill model's own first-order
  condition at a zero price of inventory risk. Both are functions of model inputs only.

### Results

PPO recovers 91.0% +/- 2.4% (worst seed 87.6%) of the optimal policy's improvement over a tuned constant quote.

Equity setting, variance reduced reward:

| strategy | mean PnL (USD per 100 share lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dp_optimal` exact optimum (dynamic program) | 5832.6 | +-32.3 | 7.90 | 0.993 | n/a | 0.89 | 80.2 |
| `hjb_optimal` HJB solver policy | 5802.7 | +-31.3 | 8.13 | 0.974 | -29.9 +- 11.2 | 0.94 | 88.4 |
| `closed_form_cartea_jaimungal` textbook closed form | 5801.8 | +-31.3 | 8.12 | 0.973 | -30.8 +- 11.2 | 0.96 | 88.4 |
| `avellaneda_stoikov` Avellaneda-Stoikov heuristic | 5494.2 | +-29.2 | 8.25 | 0.775 | -338.4 +- 21.1 | 2.28 | 85.3 |
| `rl` RL, PPO, 5-seed mean | 5703.9 | +-31.2 | 8.02 | 0.910 | -128.7 +- 10.0 | 1.84 | 77.8 |
| `rl_best_seed` RL, best of the 5 seeds | 5748.1 | +-34.9 | 7.21 | 0.939 | -84.5 +- 14.7 | 1.84 | 77.8 |
| `rl_worst_seed` RL, worst of the 5 seeds | 5650.7 | +-32.8 | 7.54 | 0.876 | -181.9 +- 18.5 | 1.78 | 69.6 |
| `naive_tuned` tuned constant quote (baseline) | 4294.9 | +-103.5 | 1.82 | 0.000 | -1537.7 +- 97.2 | 5.89 | 66.2 |
| `constant_deep` deep fixed quote | 2290.8 | +-43.0 | 2.34 | -1.295 | -3541.8 +- 44.9 | 2.89 | 13.5 |

The Avellaneda-Stoikov heuristic approximates the optimum of a different objective, exponential
utility rather than the quadratic inventory penalty used here, so its lower score is not a case
of RL beating it.

PPO recovers 56.6% +/- 34.2% (worst seed -0.7%) of the optimal policy's improvement over a tuned constant quote.

Credit setting, variance reduced reward:

| strategy | mean PnL (USD per 1mm face lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dp_optimal` exact optimum (dynamic program) | 4751.3 | +-185.8 | 1.12 | 0.977 | n/a | 0.23 | 5.4 |
| `hjb_optimal` HJB solver policy | 4754.1 | +-185.5 | 1.12 | 0.977 | +2.8 +- 20.4 | 0.24 | 5.5 |
| `rl` RL, PPO, 5-seed mean | 2751.6 | +-112.1 | 1.08 | 0.566 | -1999.7 +- 119.7 | 0.31 | 3.5 |
| `rl_best_seed` RL, best of the 5 seeds | 4203.1 | +-170.3 | 1.08 | 0.864 | -548.2 +- 120.2 | 0.29 | 5.3 |
| `rl_worst_seed` RL, worst of the 5 seeds | -33.3 | +-18.5 | -0.08 | -0.007 | -4784.6 +- 186.7 | 0.02 | 0.0 |
| `naive_tuned` tuned constant quote (baseline) | -0.8 | +-13.5 | -0.00 | 0.000 | -4752.2 +- 186.2 | 0.02 | 0.0 |
| `constant_deep` deep fixed quote | 5.3 | +-7.7 | 0.03 | 0.001 | -4746.0 +- 185.7 | 0.01 | 0.0 |

In the credit setting a tuned constant quote earns roughly nothing after inventory costs, so efficiency here is the share of the optimal policy's profit that RL captures.

RL settings, applied identically to both settings: PPO with a 64 by 64 MLP, `n_steps` 512,
`batch_size` 256, `n_epochs` 10, learning rate 3e-4, `gamma` 1.0, `gae_lambda` 0.95,
`ent_coef` 0.001, `log_std_init` -1.0, 500,000 timesteps per seed and five seeds.

The table below isolates one design choice: replacing the realised marking term with its
conditional mean during training. The removed term is a martingale increment independent of the
fills, so its expectation is zero under every policy and it cannot move the optimum, but it is
larger than the spread earned per step, which is what makes it worth removing where the spread
is small next to the price risk of a position. In this calibration the effect is modest and
uniform: the 5-seed mean efficiency moves from 0.875 to 0.910 for equity and from 0.530 to 0.566
for credit. In an earlier, much wider-spread credit calibration the same change moved credit from
0.397 to 0.812, which bounds how much of the effect belongs to reward noise in general.

Equity setting, fully realised reward:

| strategy | mean PnL (USD per 100 share lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rl` RL, PPO, 5-seed mean | 5649.9 | +-30.8 | 8.03 | 0.875 | n/a | 1.80 | 78.0 |

Credit setting, fully realised reward:

| strategy | mean PnL (USD per 1mm face lot per day) | 95% CI | PnL / std | efficiency | paired minus optimum | mean abs end of day q | fills per day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rl` RL, PPO, 5-seed mean | 2575.8 | +-110.3 | 1.02 | 0.530 | n/a | 0.34 | 5.9 |

Figures are written to `figures/`: optimal depths against inventory for both solvers, end of day
inventory distributions, learning curves against the optimum, a sample episode and the
efficiency bars.

### Parameters, with units and provenance

PARAMETER_TABLE

Anything marked ASSUMED is a choice rather than a measurement. Two of them carry the most
weight: the credit spread volatility of 4 bp per day, and the 10 requests per side per day.

### Design choices, and the honest list

* **The RL action space is `[0, d_max]^2`, and `d_max` is tight on purpose.** The unconstrained
  optimum wants very deep quotes at the inventory limit near the close. Solving the same problem
  with a bound four times wider changes the optimum by less than 0.0001 percent for
  equity and 0.0039 percent for credit, so the bound is not the binding constraint on
  the answer, but it does make exploration cover the region that matters. Both numbers are
  asserted in `tests/test_benchmark.py`. The agent is graded against the exact optimum **of its
  own bounded problem**.
* **The mid price is a martingale, so it cannot move the expected objective, only the dispersion.**
  With fills independent of the price path and a policy that does not condition on the price, the
  level of `S` enters the objective linearly and cancels. A test checks that the optimum is
  unchanged when `sigma` is set to zero.
* **At most one fill per side per step.** The one-step fill probability is the exact probability
  of at least one Poisson event, which understates the fill count by `O((lam f dt)^2)` relative to
  an event-driven simulation. At the equity parameters that is about 11 percent of fills at the
  open.
* **The observation is `(t / T, q / Q)`.** That is not a shortcut: the optimal policy is a
  function of time and inventory only, in both settings, because the mid price enters the value
  function linearly. The agent therefore has every feature the optimum uses, and no strategy is
  given a feature another does not get.
* **What the credit setting cannot show.** With about 5.4 fills per day the learning
  signal is sparse, so what the credit result measures is how much of the available edge a plain
  PPO recovers from a few thousand episodes, not what an optimised implementation could do.
* **No data.** Both settings are synthetic models with constant volatility and no adverse
  selection: fills depend only on the distance of the quote, so there is no informed counterparty
  and no queue position. The benchmark measures whether RL can learn a known policy, not whether
  that policy would make money in a real market.

### Reproduce

```bash
pip install -r benchmark/requirements.txt
python -m benchmark.run_all --ablation --save-models   # full run, about 35 minutes
python -m benchmark.run_all --quick                    # smoke run, about a minute
python check_readme.py                                 # README against the recorded results
pytest benchmark/tests                                 # 36 tests
```

`benchmark/results/summary.json` holds every number that produced the tables above, including the
per-seed efficiencies, the tuning sweep and the paired tests.

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
