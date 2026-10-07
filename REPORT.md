# Phase 3 report: reproduction-ready codebase and improvement plan

Branch `reproduction-ready`, commit `a8a06cf` on top of upstream `69d9e1e`.
Paper reference: "Market Making with Deep Reinforcement Learning from Limit Order Books",
IJCNN 2023, arXiv:2305.15821 (full text read from the arXiv HTML). Where the paper is silent
this report says UNRESOLVED rather than guessing.

Verification environment used for every claim below: macOS arm64, Python 3.10 venv with
tensorflow-macos 2.11.0, tensorforce 0.6.5 installed with `--no-deps`, pandas 2.3.3,
numpy 1.26.4. Commands and outputs are quoted inline.

---------------------------------------------------------------------------------------------------

## 1. Summary table

Bug IDs: B1 to B7 are the fatal blockers from the handoff, S1 to S11 the silent correctness
bugs, N1 to N9 findings added during implementation.

| # | File | Change | Reason | ID |
| --- | --- | --- | --- | --- |
| 1 | `main.py` | `save: bool = False,` becomes `save: bool = False` | Trailing comma made the default the truthy tuple `(False,)`, so `save_agent` always ran and crashed | B5 |
| 2 | `main.py` | `keras_model_dir` added to `TrainConfig` | The name was never defined, `NameError` on any run reaching `save_agent` or `--load` | B6 |
| 3 | `main.py` | `agent.save(..., filename=config['agent_save_filename'])` | Passed the agent object as the filename: `TypeError: join() argument must be str` | B7 |
| 4 | `main.py` | `save_keras_model` helper passes `save_format='tf'` only when the signature accepts it | Explicit format, and works on both the TF 2.6 and TF 2.11 Keras versions | B7 |
| 5 | `main.py` | `train_days`, `test_days`, `num_step_per_episode`, `n_train_loop`, `max_episode_timesteps` moved into `TrainConfig` | They were module globals defined only under `if __name__ == '__main__'`, so the module was not importable and the episode cap was hardcoded at 1000 while episodes were 2000 | S9, prompt 1.3 |
| 6 | `main.py` | `save_run_results` writes `config.json`, `train_episodes.csv`, `test_episodes.csv`, `test_daily.csv` to `./results/{run_id}/` | A successful run produced no artifact at all | S10, prompt 1.6 |
| 7 | `main.py` | `train_deterministic` (default False) and `test_deterministic` (default True) config fields, passed to `agent.act` | `Agent.act` defaults `deterministic=True` and TensorForce skips exploration entirely when it is true, so the shipped loop trained on the greedy action and only 2 of 8 discrete actions were ever emitted | N1 |
| 8 | `main.py` | `exploration` config field forwarded to the agent | TensorForce defaults exploration to 0.0, i.e. no exploration | N2 |
| 9 | `main.py` | `build_lob_encoder` raises `FileNotFoundError` with the fix instructions when the pretrained checkpoint is missing | Replaced an opaque `NotFoundError` from `load_weights` | B2 |
| 10 | `main.py` | dead `data_collector` accumulation removed | It retained every episode forever, unbounded memory growth with no reader | N9 |
| 11 | `main.py` | `learning_rate: int = 1e-4` becomes `float` | Wrong annotation on a float field | n/a |
| 12 | `environment/base_env.py` | `self.is_trade.loc[list(set(...))]` | `df.loc[set(...)]` raises on pandas 2, so the environment could not be constructed at all | B4 |
| 13 | `environment/base_env.py` | `raw_dir` parameter replaces hardcoded `raw/` paths | Tests and adapters could not point at another dataset root | N7 |
| 14 | `environment/base_env.py` | `_prepare_episode` returns `None` for windows with fewer than two usable trade ticks or shorter than T, `reset_seq` and `reset_random` share it | The old code raised `StopIteration` on quiet windows and produced a wrongly shaped LOB state for short windows | S8, prompt 1.6 |
| 15 | `environment/base_env.py` | first usable decision index is `T + latency - 1` instead of `T` | Generalises the filter to latency greater than 1 while matching the old behaviour at latency 1 | S6 |
| 16 | `environment/base_env.py` | `match` and `close_position` use `get_price_info(self.i - self.latency)`, `match` fills against trades in `(t_1, i]` | Unified the timing reference across decision, quote placement and fill check | S6, prompt 1.5 |
| 17 | `environment/base_env.py` | `update_agent` splits value change into `trading_pnl_total` and `holding_pnl_total`, `volume` counts both sides, `traded_units` added | Holding PnL was never populated (always 0) and volume counted buys only | S7, prompt 1.6 |
| 18 | `environment/base_env.py` | `profit_ratio` is PnL / volume, `nd_pnl` and `pnl_map` return nan instead of dividing by zero | Paper section IV-C4 defines Profit Ratio as PnL over volume; the code used total value | S7 |
| 19 | `environment/base_env.py` | `get_final_result` exposes the new components | PnL attribution and reward diagnostics | prompt 1.6 |
| 20 | `environment/env_continuous.py` | `reward_mode` in `{composite, pnl}`, default `composite`; real composite `r_ma*TP + r_da*DP - r_ip*IP` | The shipped reward was plain PnL with the composite expression commented out, which made every `wo_*` reward ablation inert | S3, prompt 1.5 |
| 21 | `environment/env_continuous.py` | action bounds `[0, 1]` | The spec declared `[-1, 1]` while the implementation assumed `[0, 1]`, allowing negative spreads and crossed quotes | prompt 1.5 |
| 22 | `environment/env_continuous.py` | `max_bias`, `max_spread`, `inventory_penalty`, `asymmetry_eta`, `spread_penalty_lambda` config fields | Paper values 0.05, 0.1, 0.01, 0.5; the shipped continuous env used eta 0.9 | prompt 1.5 |
| 23 | `environment/env_continuous.py` | `action2order` docstring with the latency convention and a worked example | Documents what "latency = 1" means for decision, quotes and fills | prompt 1.5 |
| 24 | `environment/env_discrete.py` | `num_actions` config field, default 8, validated to 1..8 | Declared 5 while implementing 0..7, so the flatten action was unreachable | S4, prompt 1.5 |
| 25 | `environment/env_discrete.py` | `reward_mode` and the paper's L2 inventory punishment | Same reward problem as the continuous env | S3 |
| 26 | `network/network.py` | `dense_input.append(market_state)` in the market-state branch | It appended `agent_state` twice, so the market state was never consumed and the agent state was dropped | S1 |
| 27 | `network/network.py` | `get_model(lob_model, T, state_keys)` builds inputs from `state_spec` | The env emitted lob, market, agent while the network expected lob, agent, market | S2 |
| 28 | `network/network.py` | FC-LOB Dense layers chained | All three read the flattened input, so the 1024 and 256 layers were outside the graph and the ablation was one linear layer | S5 |
| 29 | `network/network.py` | `make_compute_output_shape(latent_dim)` factory, `LOB_LATENT_DIM` constant | `compute_output_shape` hardcoded 64 and the monkeypatch is retained on purpose | prompt 1.5 |
| 30 | `state_spec.py` | new: `STATE_KEYS`, `active_state_keys`, `state_specs`, `state_input_dims`, `LOB_COLUMNS` | One source of truth for the state order and the LOB channel order | S2, S11, prompt 1.4 |
| 31 | `utils.py` | `lob_norm` reindexes to `LOB_COLUMNS` and guards against an all-zero volume level | Channel order was whatever the CSV contained, and division by a zero maximum produced inf/nan which TensorForce rejects | S11, N9 |
| 32 | `utils.py` | `load_data(..., data_dir='data')` | The data root was hardcoded, so nothing could point at a fixture or another dataset | N7 |
| 33 | `compat.py` | new: Keras optimizer shim for TF >= 2.11, no-op below | TensorForce 0.6.5 calls `_create_all_weights` and `_create_hypers`, removed in TF 2.11 | B3, prompt 1.2 |
| 34 | `agent/tensorforce_agent.py` | applies the shim on import, adds `eager_mode` and `exploration` pass-through | Guarantees the shim on every agent creation path; `eager_mode` is required for the wiring probe | N1, prompt 1.2, 1.4 |
| 35 | `pretrain.py` | new: trains the 3-class classifier and writes `./ckpt/pretrain_model_{code}/weights` | The script was missing from the repository | B2, prompt 1.6 |
| 36 | `data/adapter.py` | new: schema documentation, `validate_dataset`, `write_day`, `generate_synthetic`, documented `convert_tardis` skeleton | Nothing documented the on-disk schema, and there was no way to exercise the pipeline without the private dataset | B1, prompt 1.6 |
| 37 | `tests/` | new: 17 tests over wiring, reward modes, identity, guards, action space | Prompt 1.4 and 1.7 | prompt 1.4 |
| 38 | `requirements-macos.txt` | new | The shipped `conda_setup.yaml` cannot be recreated on arm64 | B3, prompt 1.1 |
| 39 | `Dockerfile` | new, linux/amd64, TF 2.6.0 with corrected 2.6.x companion pins | Faithful stack for reproducing the original environment | B3, prompt 1.1 |
| 40 | `README.md` | rewritten | The original was three lines with no setup, data or run instructions | prompt 1.6 |
| 41 | `.gitignore`, `pytest.ini` | new | Keep datasets and checkpoints out of the repository | prompt 1.6 |

---------------------------------------------------------------------------------------------------

## 2. Code

Unified diffs: `git show a8a06cf` for the modified files and `git diff 69d9e1e a8a06cf`
for the whole change. New files, full contents in the repository:

| File | Lines | Contents |
| --- | --- | --- |
| `state_spec.py` | 107 | state key order, dims, `LOB_COLUMNS`, validation helpers |
| `compat.py` | 102 | feature-probed optimizer shim plus the audit of the patched API surface |
| `pretrain.py` | 140 | 3-class trend classifier training, checkpoint writer, session-window guard |
| `data/adapter.py` | 337 | schema constants, `validate_dataset`, `write_day`, `generate_synthetic`, `convert_tardis` |
| `tests/conftest.py` | 74 | dataset fixtures and shared env factory |
| `tests/test_wiring.py` | 211 | structural, sensitivity and end-to-end sentinel wiring checks |
| `tests/test_smoke.py` | 188 | episode loop, PnL identity, reward modes, action space, guards, pretrain pipeline |
| `requirements-macos.txt` | 45 | arm64 pins plus the exact install order |
| `Dockerfile` | 68 | TF 2.6.0 linux/amd64 image |
| `README.md` | 186 | honest setup, data and run instructions |
| `.gitignore`, `pytest.ini` | 15 | ignore rules and pytest config |

The most important hunk in the whole change, `network/network.py`:

```diff
-    if with_agent_state:
-        agent_state = keras.layers.Input(shape=(24,))
-        input_ls.append(agent_state)
-        dense_input.append(agent_state)
-    else:
-         print('w/o agent state!')
-
-    if with_market_state:
-        market_state = keras.layers.Input(shape=(24,))
-        input_ls.append(market_state)
-        dense_input.append(agent_state)          # <-- market state never reached the concat
+    keys = check_canonical_order(STATE_KEYS if state_keys is None else state_keys)
+    dims = state_input_dims(T, keys=keys, market_state_dim=..., agent_state_dim=...)
+    input_ls = list()
+    dense_input = list()
+    for key in keys:
+        state_input = keras.layers.Input(shape=dims[key], name=key)
+        input_ls.append(state_input)
+        if key == 'lob_state':
+            dense_input.append(lob_model(state_input))
+        else:
+            dense_input.append(state_input)
```

Measured effect of that bug, from the pre-fix probe of the built graph:

```
internal concat layer inputs: ['model/flatten/Reshape:0', 'input_3', 'input_3']
changing 2nd positional input (agent slot) changes output : True
changing 3rd positional input (market slot) changes output: False
```

---------------------------------------------------------------------------------------------------

## 3. Run instructions

### macOS arm64

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
pip install --no-deps tensorforce==0.6.5
pip install pytest==9.1.1
python -c "import tensorflow, tensorforce; print(tensorflow.__version__, tensorforce.__version__)"
# 2.11.0 0.6.5

python -m data.adapter --synthetic --code 000001 --day 20191101 --rows 4000
python pretrain.py --code 000001 --days "['20191101']" --epochs 20
python main.py --code 000001 --train_days "['20191101']" --test_days "['20191101']" \
    --num_step_per_episode 400 --n_train_loop 1 --save True
pytest
```

`tensorforce` must be installed with `--no-deps`: its metadata pins `tensorflow==2.6.0`,
`numpy==1.19.5` and `h5py~=3.1.0`, none of which resolve on arm64. `pip check` will report
those as unsatisfied, which is expected and documented in `requirements-macos.txt`. numpy must
stay below 2 or TensorFlow 2.11 aborts at import with a `_pywrap_bfloat16` TypeError, which was
hit and fixed during this work.

### Linux x86_64

```bash
docker build --platform linux/amd64 -t mm-lob:tf26 .
docker run --rm -it --platform linux/amd64 -v "$PWD:/workspace" mm-lob:tf26 bash
# inside
python -m data.adapter --synthetic --rows 4000
python pretrain.py --days "['20191101']"
python main.py --train_days "['20191101']" --test_days "['20191101']" --save True
```

TF 2.6 x86 wheels need AVX. Docker Desktop on Apple Silicon does not provide it, so this image
is for a real x86_64 Linux host. The Dockerfile corrects the original pin set, which paired
`tensorflow==2.6.0` with `tensorflow-estimator==2.10.0` and `tensorboard==2.10.1`.

### Verified acceptance evidence

```
$ pytest -q
17 passed

$ python pretrain.py --days "['20191101']" --epochs 1
label distribution (1=up, 2=stationary, 3=down):
1    0.395760
2    0.111055
3    0.493185
pretraining samples: X (1932, 50, 40, 1), Y (1932, 3)
Encoder weights saved to ./ckpt/pretrain_model_000001/weights

$ python main.py --train_days "['20191101']" --test_days "['20191101']" \
      --num_step_per_episode 400 --n_train_loop 1 --save True
Run id: 20261007_143039
Saved run outputs to ./results/20261007_143039
             PnL      ND-PnL  PnL-MAP  ... holding_pnl  trading_pnl  episode_reward
2019-11-01  49.0  828.461734    1.334  ...         6.0         43.0           64.65
# ckpt/agent_model (Keras SavedModel) and ckpt/agent/{agent.json,agent.npz} written
# holding_pnl + trading_pnl == PnL exactly

$ python main.py --env_type discrete --agent_type dueling_dqn --exploration 0.5 ...
2019-11-01  345.0  10988.63014  ...        324.0          391.84

$ python main.py --load True --agent_load_dir ./ckpt/agent ...
2019-11-01  51.0  862.670564  ...         42.0           65.13
```

The discrete coverage criterion is enforced by
`tests/test_smoke.py::test_discrete_agent_emits_every_configured_action`, which asserts that
all 8 actions are emitted. It only passes with `exploration` set and `deterministic=False`,
which is the N1 finding below.

---------------------------------------------------------------------------------------------------

## 4. Paper-fidelity report

Values below are resolved against arXiv:2305.15821. Citation labels are the paper's own
section and equation numbers.

| Decision | Paper | Shipped code | Resolution | Confidence |
| --- | --- | --- | --- | --- |
| Reward function | Section III-D, Eq. (12) to (16): `R = DP + TP - IP`, dampened PnL `DP = dPnL - max(0, eta*dPnL)` Eq. (13), matched PnL Eq. (14), L2 inventory punishment Eq. (15) | `reward = pnl - spread_punishment`, composite expression commented out, and an undocumented spread penalty applied only while flat | `reward_mode='composite'` is now the default and implements Eq. (16) exactly, with `r_ma`, `r_da`, `r_ip` driven by the `wo_*` flags. `reward_mode='pnl'` preserves the shipped behaviour, and `spread_penalty_lambda=0` gives a clean plain-PnL baseline | paper HIGH, code HIGH |
| eta (asymmetry) | 0.5, section IV-C2 | 0.9 in the continuous env, 0.5 in the discrete env | default 0.5 for both, configurable | HIGH |
| zeta (inventory) | 0.01, section IV-C2 | 0.01 in both | unchanged, configurable | HIGH |
| Spread penalty | Not in Eq. (16) | continuous env only | kept only under `reward_mode='pnl'`, documented as non-paper | HIGH |
| Discrete action space | Section III-C1: 8 actions, action 7 closes the position with market orders | `num_values=5` while `action2order` implemented 0 to 7 | `discrete_num_actions` default 8 | HIGH |
| Continuous action bounds | Section III-C2, Eq. (8) to (10): A1, A2 in [0, 1] | declared [-1, 1], implemented as [0, 1] | bounds are now [0, 1] | HIGH |
| Quote scaling | Section IV-C2: max_bias 0.05, max_spread 0.1 | same values inline | config fields, unchanged defaults | HIGH |
| Inventory cap omega | Section IV-C2: omega = 10, quoting is blocked in a direction beyond `|position| > omega * minimum_trade_unit` | `if self.inventory < -10*TRADE_UNIT` in `base_env.execute` | unchanged, with `TRADE_UNIT` = 100 shares, so the cap is 1000 shares | HIGH |
| LOB channel order | Eq. (1): `{P_ask_i, V_ask_i, P_bid_i, V_bid_i}`, window 50 x 40 in Table I | whatever column order the CSV had; the unused `utils.reorder` documents a volume-first order instead | `state_spec.LOB_COLUMNS` is price-first per level and is applied in both the RL path and the pretraining path. The paper's Eq. (1) wins over the stale helper | HIGH for the paper, order now pinned in code |
| n (book levels) | never printed as a number | 10 per side | 10, from Table I's width of 40 | INFERRED |
| Market state | Section III-A, Eq. (2) to (4): 18 order strength features (10s/60s/300s) plus 3 realised volatility and 3 RSI (5/10/30 min) = 24 | `_get_market_state` returns 6, `_get_order_strength_index` returns 18 | unchanged, matches | HIGH |
| Agent state | Section III-A3: normalised inventory plus a time factor, current time over total time | `[inventory/(10*TRADE_UNIT)]*12 + [t/episode_length]*12` | unchanged, matches the paper's wording | HIGH |
| FC-LOB ablation | Section IV-B2 lists hidden sizes 1024, 256, 64, and 3 is the classifier head | all three Dense layers read the flattened input, so 1024 and 256 were absent from the graph | chained 1024 to 256 to 64, sizes taken from the paper | HIGH |
| Attn-LOB internals | Not stated (deferred to a figure and references) | conv and inception blocks, 10 heads, key_dim 16, latent 64 | kept as shipped, latent dim parameterised. These numbers cannot be credited to the paper | UNRESOLVED in the paper |
| Pretraining labels | Section IV-B1: 3 classes, horizon k = 10, threshold alpha = 1e-5, T = 50 | `getLabel` matches | `pretrain.py` uses the same values as defaults | HIGH |
| Pretraining loss | Not stated | `Dense(3, softmax)` plus one-hot labels implies categorical cross-entropy | categorical cross-entropy, marked inferred | INFERRED |
| Encoder attachment | "applied this model as a function approximator", section III-A1 | `model_pretrain.layers[1]` | unchanged | HIGH |
| Learning rate, batch size, discount, memory, pretrain epochs and loss | Not stated anywhere in the text | lr 1e-4, batch 32, discount 0.99, memory 200000 | kept, epoch/optimizer values in `pretrain.py` marked `# ASSUMED` | UNRESOLVED |
| Episode length | Section IV-C2: 2000 events | 2000 order book rows, of which only trade ticks become decisions | kept as shipped (`num_step_per_episode`), but the decision count per episode is lower than 2000. Flagged, not changed, because redefining the episode changes the method | RESOLVED as shipped, gap flagged |
| Train/test split | Section IV-B1: 21 trading days in November 2019, roughly first half train and second half test; exact dates not printed | 8 train days and 13 test days | kept the shipped lists, now config fields. Documented as a divergence from the described split | UNRESOLVED |
| Acting during training | Nothing stated | `agent.act(independent=True)` with the default `deterministic=True` | see N1. Now `train_deterministic=False`, `test_deterministic=True`, with the original behaviour one flag away | UNRESOLVED in the paper |
| Metrics | Section IV-C4: ND-PnL is PnL over average spread, PnL-MAP is PnL over mean absolute position, Profit Ratio is PnL over total trading volume. Sharpe described qualitatively only, no formula | ND-PnL and PnL-MAP match, Profit Ratio used total value not PnL | Profit Ratio now uses PnL over traded notional, all three guarded against zero denominators. The paper says "total trading volume" without specifying units; notional (CNY) is assumed because it makes the ratio dimensionless, if the authors meant shares the value scales by the average price | paper HIGH, units ASSUMED, Sharpe UNRESOLVED |
| Data source | Section IV-A: Shenzhen Stock Exchange, November 2019, 21 days; the paper does not name the vendor and does not state that the data is public | `raw/GTA_SZL2_*`, i.e. CSMAR (Guo Tai An) Shenzhen level-2 | not obtainable from the paper or the repository. See section 6 | HIGH |

### Findings added during implementation

* **N1, exploration is dead in the shipped loop.** `Agent.act` defaults `deterministic=True`
  and TensorForce guards exploration with
  `tf.cond(pred=deterministic OR exploration == 0, true_fn=greedy, false_fn=apply_exploration)`
  in `core/models/tensorforce.py`. The shipped `train_a_day` called
  `agent.act(states=states, independent=True)`, so exploration was skipped in every training
  step, for both PPO and DQN. Measured before the fix: the dueling DQN emitted only action 0
  and action 1 over a full episode even with `exploration=1.0` set, because `deterministic`
  short-circuits it. Now configurable, defaults flipped to the standard
  sample-while-training, greedy-while-testing.
* **N2, TensorForce's default exploration is 0.0** for both agents ("default: no exploration"
  in the 0.6.5 docstrings). Fixed only by making it configurable.
* **N3, the pretraining and RL paths normalise volumes differently.** `utils.process_data`
  divides each volume level by the maximum over the whole loaded dataset; `utils.lob_norm`
  divides by the maximum inside the T-row window. Price channels agree (both divide by the
  row's own mid). Pretrained encoder weights therefore see a different volume scale than the
  RL state, which is a fidelity risk for the `--wo_pretrain` comparison. Not changed, because
  fixing it means choosing one constant for both paths and re-running pretraining. Flagged in
  `pretrain.py`.
* **N4, the pretraining window is narrower than the environment's.** `utils.process_data`
  keeps only `10:00:00 < time < 14:30:00`, while `load_orderbook` accepts 09:30:00 to
  14:57:00. A dataset covering only the open silently yields zero pretraining samples
  (observed: `X (0, 40)`, then `ValueError: negative dimensions are not allowed` inside
  `data_classification`). `pretrain.py` now raises a clear error and the synthetic generator
  starts at 10:00:00.
* **N5, `utils.reorder` contradicts the paper.** Its docstring specifies a volume-first
  channel order; the paper's Eq. (1) is price-first. The helper is dead code, and the channel
  order is now pinned in `state_spec.LOB_COLUMNS`, matching the paper.
* **N6, the numpy checkpoint is dominated by the replay memory.** Measured on the acceptance
  run: `ckpt/agent/agent.npz` was 543 MB, of which
  `memory/states_lob_state-buffer.npy` was 528 MB, a (66000, 50, 40, 1) float32 buffer.
  `agent.save(format='numpy')` serialises experience memory along with the variables. The
  Keras SavedModel in `ckpt/agent_model` already holds the network, so persisting the buffer is
  pure waste for this workflow.
* **N7, hardcoded data roots.** `load_order` and `load_trade` read `raw/...` relative to the
  process working directory and `utils.load_data` read `data/...`, so no test, fixture or
  adapter could point anywhere else. Both are parameters now.
* **N8, `agent.json` is not self-contained.** `Agent.save` warned that "Some agent argument
  could not be encoded to JSON", because the network is a Keras model object rather than a
  spec. Loading therefore requires the Keras model first, which is exactly what the `--load`
  path does; verified working.
* **N9, unbounded memory in the training loop.** `data_collector` in `train_a_day` appended
  every episode's states, actions, terminals and rewards to a process-lifetime list that
  nothing read. Removed.

---------------------------------------------------------------------------------------------------

## 5. Phase 2 improvement plan

Ranked by expected value for a researcher who first wants a credible reproduction and then a
publishable or deployable extension. "Baseline" means the Phase 1 code on this branch.

### 1. Evaluation rigor: walk-forward splits, seeds, PnL decomposition, risk metrics

* **Axis:** evaluation rigor.
* **Hypothesis:** the current measurement is a single pass over a handful of days with one
  seed, and it reports PnL without an error bar. Every conclusion drawn from it is
  unsupported. With 13 test days and a per-day PnL standard deviation of the same order as the
  mean, a daily-mean PnL is indistinguishable from zero at any conventional confidence level.
* **Change:** (a) multiple seeds, 5 minimum, reporting mean and a 95 percent CI of the daily
  mean PnL; (b) walk-forward: train on days 1 to k, test on k+1, roll forward, instead of one
  fixed split; (c) report the PnL decomposition already available in the new
  `holding_pnl`/`trading_pnl` columns, plus Sharpe and Sortino computed from per-day returns,
  maximum drawdown of the equity curve, and inventory statistics; (d) report fill rate and the
  quote-to-trade ratio.
* **Worked example:** from the verified acceptance run, `PnL = 49.0` with
  `holding_pnl = 6.0` and `trading_pnl = 43.0`. The decomposition says 88 percent of the PnL is
  spread capture and 12 percent is inventory drift. A single PnL number cannot distinguish a
  market maker earning the spread from one that got lucky with inventory, and only the second
  is fragile.
* **How to test:** run the same agent over the same days with 5 seeds, then check that the
  reported CI contains the mean of the seeds and that the PnL component sums equal PnL for
  every episode (the invariant is already asserted in `tests/test_smoke.py`).
* **Effect size:** no direct PnL effect, but it is the difference between a defensible claim
  and an anecdote. **Confidence: high.** **Cost:** 1 to 2 days engineering, 5x the current
  compute per experiment.

### 2. Baselines: Avellaneda-Stoikov, Gueant-Lehalle-Fernandez-Tapia, fixed and inventory-skew quoters

* **Axis:** baselines the paper may lack.
* **Hypothesis:** the paper's own baseline set is thin for a reader trying to judge the RL
  contribution, and without a classical baseline no PnL number means anything.
* **Change:** implement four quoters against the same `execute` API: (i) fixed symmetric
  spread at 1 tick; (ii) fixed quoting at level 1 to 3; (iii) an inventory-skew heuristic that
  shifts the reservation price by `k * inventory`; (iv) Avellaneda-Stoikov with the paper's
  Eq. (17) and (18), `r(s,q,t) = s - q*gamma*sigma^2*(T-t)` and
  `delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa)`, calibrating
  `gamma` and `kappa` from the training data. Reference 9 in the paper is the A-S model, so
  this is a direct comparison, not an invention.
* **Worked example:** with `sigma` estimated at 1e-4 per event, `T - t = 2000` events,
  `gamma = 0.1` and `kappa = 1.5`, the half-spread is
  `0.5*[0.1*1e-8*2000 + (2/0.1)*ln(1 + 0.1/1.5)] = 0.5*[2e-6 + 20*0.0645] = 0.645`. The A-S
  quote would then be about 64 ticks wide on a 0.01 grid, which on a Shenzhen stock is
  uncompetitive, and that is itself the point: it shows the RL agent's advantage is
  in the adaptive spread, not in the theory.
* **How to test:** same days, same seeds, same fills model, reported in the same table as the
  agents. The claim to check is that A-S and the RL agent differ in average spread and in
  inventory variance, not only in PnL.
* **Effect size:** makes or breaks the paper's story. **Confidence: high** that the comparison
  is necessary, medium on the sign of the result. **Cost:** 3 to 5 days.

### 3. Simulator realism: fees, rebates, tick and lot, partial fills, endogenous impact

* **Axis:** simulator realism.
* **Hypothesis:** the fill model is optimistic and cost-free, so absolute PnL is overstated,
  and any strategy that ignores costs will over-trade. This is the single largest threat to
  external validity.
* **Change:** add a fee model (per-side commission plus any sell-side stamp duty, or a
  maker/taker schedule for crypto), make fills partial instead of all or nothing, separate
  quoted size from filled size, and add an optional adverse-selection term that removes a
  fraction of the fills that occur when the mid moves against the quote.
* **Worked example (fees):** one unit is 100 shares at 10.00, i.e. 1000 CNY notional. With a
  1 bp commission per side plus a 5 bp sell-side stamp duty, a round trip costs
  `1000*0.0001 + 1000*0.0001 + 1000*0.0005 = 0.1 + 0.1 + 0.5 = 0.7` CNY. The verified
  acceptance run made `49.0` CNY over `4400` shares, i.e. `49/44 = 1.11` CNY per 100-share
  unit. Netting the 0.7 CNY cost leaves 0.41 CNY per unit, a 63 percent reduction. Both
  numbers are illustrative, the cost is a stated assumption and the PnL is from synthetic
  data, but the sensitivity is the point: a strategy at 1.11 CNY per unit gross is fragile.
* **How to test:** re-run the acceptance command with fees enabled and compare gross to net
  PnL, plus the change in the quote-to-trade ratio. A strategy whose PnL collapses when fees
  are added was never viable.
* **Effect size:** large and negative on gross PnL, which is the honest number.
  **Confidence: high.** **Cost:** 2 to 4 days.

### 4. Port off TensorForce to a maintained stack

* **Axis:** model and algorithm.
* **Hypothesis:** TensorForce 0.6.5 is unmaintained, pins TensorFlow 2.6, and needs a
  compatibility shim (compat.py) to run on anything newer. It also silently skips exploration
  (N1) and saves 528 MB of replay buffer per checkpoint (N6). Its cost now exceeds its value.
* **Change:** reimplement the agents in PyTorch with CleanRL or Stable-Baselines3 or TorchRL,
  keeping the same environment API. SAC is the better fit for the continuous action than PPO
  because quoting is a continuous control problem with a smooth objective, and PPO's
  on-policy requirement wastes the expensive simulator. A distributional head (QR-DQN or
  C51) or a CVaR objective is the natural next step for tail risk.
* **Worked example:** the current checkpoint cost is measurable. TensorForce writes 543 MB per
  save, of which 528 MB is the LOB replay buffer at (66000, 50, 40, 1) float32. A PyTorch
  implementation that checkpoints only the policy would write roughly 1 to 2 MB and would make
  a 5-seed sweep feasible on the same disk.
* **How to test:** match the Phase 1 baseline PnL decomposition day by day within noise, then
  compare sample efficiency (updates to reach a given PnL) and wall-clock per episode.
* **Effect size:** medium on PnL, large on velocity of iteration. **Confidence: high** that
  the port pays for itself, medium on a PnL improvement. **Cost:** 1 to 2 weeks.

### 5. Exploration and optimization hygiene

* **Axis:** model and algorithm.
* **Hypothesis:** N1 and N2 show the shipped loop trained with zero exploration for both
  agents. Even with the flag fixed, the defaults (fixed epsilon, no entropy tuning, no reward
  normalisation) leave learning unstable.
* **Change:** an epsilon or entropy schedule with a decay tied to updates, entropy
  regularisation for PPO, reward scaling to unit variance over a rolling window, gradient
  clipping, and per-seed early stopping based on a validation day.
* **Worked example:** with 8 discrete actions and no exploration, the first update sees only
  actions 0 and 1, so Q for the other six actions never receives a gradient and the argmax can
  never leave the visited set. The measured symptom was exactly this: only actions 0 and 1
  appeared over an entire episode before the fix, and all 8 appeared once
  `deterministic=False` and `exploration=1.0` were set.
* **How to test:** the existing
  `test_discrete_agent_emits_every_configured_action` guards the discrete case; add the
  continuous analogue (action variance above a floor early in training) and a learning curve
  per seed.
* **Effect size:** potentially large for the discrete agent, where the current result is
  probably meaningless. **Confidence: high.** **Cost:** 1 to 2 days.

### 6. Reward design: mean-variance or CARA utility, terminal inventory penalty, differential Sharpe

* **Axis:** reward design.
* **Hypothesis:** the paper's Eq. (16) reward shapes the mean of PnL with a quadratic
  inventory penalty, which is a proxy for risk. A direct risk-adjusted objective should
  dominate, because it prices the variance the agent actually generates.
* **Change:** offer `reward_mode` variants: (i) mean-variance, `dPnL - lambda*Var(dPnL)`;
  (ii) CARA or exponential utility, `1 - exp(-gamma*dPnL)`; (iii) the paper's reward plus an
  explicit terminal penalty on `inventory` at the episode close, which the code currently
  handles by liquidating at the touch; (iv) differential Sharpe.
* **Worked example (the existing asymmetry already shows the mechanism):** with the paper's
  `eta = 0.5`, a +10 CNY mid-price move on held inventory contributes
  `DP = 10 - max(0, 0.5*10) = 5`, while a -10 CNY move contributes `DP = -10 - max(0, -5) = -10`.
  Equal-magnitude mark-to-market moves are weighted 1:2 against the agent, which is a hard
  coded risk aversion. A CARA utility with `gamma = 0.1` would instead give
  `1 - exp(-0.1*10) = 0.632` versus `1 - exp(0.1*10) = -1.718`, a 1:2.7 ratio that scales with
  the size of the move rather than being fixed at 2:1.
* **How to test:** compare reward variants at matched gross exposure, using the decomposition
  and drawdown metrics from item 1. The claim to check is lower inventory variance at similar
  spread capture.
* **Effect size:** medium, with a real chance of a large drawdown improvement.
  **Confidence: medium.** **Cost:** 3 to 5 days.

### 7. State features: OFI, microprice, queue imbalance, trade-sign autocorrelation

* **Axis:** state features.
* **Hypothesis:** the market state is six coarsely windowed indices (10, 60 and 300 seconds)
  plus realised volatility and RSI. It discards the most predictive microstructure signals
  available at the top of the book: order flow imbalance, the size-weighted microprice, queue
  imbalance at the touch, and the autocorrelation of trade signs.
* **Change:** add these as extra blocks with their own dims, wire them through
  `state_spec`, and ablate them against the baseline.
* **Worked example (microprice):** with `bid1 = 10.00` at 300 shares and `ask1 = 10.01` at
  900 shares, the microprice is `(10.00*900 + 10.01*300)/1200 = 10.0025`, one quarter of a
  tick above the mid of 10.005... more precisely, the microprice sits at 10.0025 versus a mid
  of 10.005, so it is 0.25 ticks below the mid and points down, while the naive mid carries no
  such information. Placing the reservation price off the microprice rather than the mid
  reduces adverse selection mechanically.
* **How to test:** ablate each block, and check both PnL and adverse selection (the fraction
  of fills followed by an adverse mid move).
* **Effect size:** medium, plausibly the largest single modelling gain after fees.
  **Confidence: medium to high.** **Cost:** 3 to 5 days plus the data plumbing to compute
  them from the order and trade feeds.

### 8. Offline RL on historical tapes

* **Axis:** offline RL.
* **Hypothesis:** the simulator is a replay of one realised path. Every episode the agent sees
  is drawn from the same tape, so an online agent effectively overfits to a single market
  realisation, and the fill model limits what it can learn. Offline RL on many tapes uses the
  data more efficiently and does not require the learned simulator to be right.
* **Change:** shape the day's data into a fixed dataset of (state, action, reward, next state)
  transitions using a behaviour policy, then run CQL or IQL. `d3rlpy` is already pinned in the
  original `conda_setup.yaml`, so the authors had this in mind.
* **Worked example:** one day of 2000 book rows at 1 Hz with a continuous action yields about
  2000 transitions from one tape. With 21 days that is roughly 42000 transitions, which is
  small but workable for IQL and hopeless for CQL with a large LOB encoder; the practical
  conclusion is that the encoder must be shared or frozen, which is exactly what the
  pretrained Attn-LOB encoder enables.
* **How to test:** offline policy selection on a held-out day, then online evaluation on a
  later day. The claim to check is that the offline agent reaches baseline PnL with fewer
  environment interactions.
* **Effect size:** unclear on PnL, large on sample efficiency. **Confidence: medium.**
  **Cost:** 1 to 2 weeks.

### 9. Pretraining objective and label quality

* **Axis:** model and algorithm.
* **Hypothesis:** the 3-class trend label with `alpha = 1e-5` over `k = 10` events produces a
  nearly balanced but very noisy target. In the acceptance run the label distribution was 39.6
  percent up, 11.1 percent stationary, 49.3 percent down, on synthetic data, and the observed
  accuracy was 0.48 with a validation accuracy of 0.34, i.e. near chance. If the pretrained
  encoder is near chance, the `--wo_pretrain` ablation has little to remove and the paper's
  claim about pretraining cannot be tested.
* **Change:** sweep `k` and `alpha` (the paper fixes them but not their sensitivity), and try
  self-supervised alternatives: masked level prediction, next-book-event prediction, or
  contrastive learning on adjacent book states. Also fix N3 so the pretraining and RL volume
  scales agree.
* **Worked example:** with 3 classes and a majority class at 49 percent, chance accuracy on the
  balanced subset is 33 percent. The measured validation accuracy of 0.335 is indistinguishable
  from chance, so on this data the pretrained encoder is not learning a usable trend signal.
* **How to test:** compare (a) pretrained, (b) random init, (c) from-scratch joint training,
  with 5 seeds and the item 1 metrics. The ablation that matters is pretrained versus random.
* **Effect size:** medium, and it directly tests a paper claim. **Confidence: medium.**
  **Cost:** 3 to 5 days.

### 10. Robustness: regime randomisation, toxic flow, cross-instrument transfer

* **Axis:** robustness.
* **Hypothesis:** the agent is trained and tested on November 2019, one regime, three
  instruments at most. Its behaviour under a volatility spike or a liquidity drought is
  unknown, and market making is precisely the strategy that must survive those.
* **Change:** domain randomisation over volatility and liquidity during training, an explicit
  toxic-flow test (a burst of adversarial order flow), and leave-one-instrument-out transfer.
* **Worked example:** the inventory punishment is `zeta*(Inv/100)^2 = 0.01*(Inv/100)^2`. At
  `Inv = 500` shares this is 0.25 per step; at `Inv = 1500` it is 2.25, nine times larger for
  three times the inventory. A regime that triples typical fill size therefore increases the
  effective penalty by an order of magnitude, which is the kind of distribution shift the
  current single-regime training never sees.
* **How to test:** train on one instrument and test on another with no fine tuning, and report
  the degradation. Any collapse is information.
* **Effect size:** unknown on average PnL, large on tail behaviour. **Confidence: medium.**
  **Cost:** 1 week, mostly data engineering.

### 11. Execution constraints and order lifecycle

* **Axis:** simulator realism and evaluation rigor.
* **Hypothesis:** the agent quotes every step with a fixed size of one `TRADE_UNIT` and never
  cancels, so the modelled order lifecycle is unrealistic and the quote-to-trade ratio is not
  measured. Real makers cancel far more than they fill, and Shenzhen rules cap order rates.
* **Change:** model cancellations and replacements explicitly, let the agent choose size, and
  track quote-to-trade ratio and average resting time. Report the ratio next to PnL, because a
  high PnL obtained by quoting 100 times more than needed is not deployable.
* **Worked example:** with a fill probability at the touch of `traded/(traded + depth)`, the
  supplied synthetic data gives 500 to 2000 shares traded against 100 to 500 shares of depth,
  so a back-of-queue fill probability of about 0.5 to 0.95. Assuming front of queue, i.e.
  probability 1, would overstate fills by up to 2x at the touch.
* **How to test:** compare fill counts and PnL with the queue model on and off. The difference
  is the value of the queue model.
* **Effect size:** large in absolute PnL, negative. **Confidence: high** that the queue model
  matters, medium on its calibration. **Cost:** 3 to 5 days.

### 12. Interpreting the reward in currency units

* **Axis:** evaluation rigor, reward design.
* **Hypothesis:** the composite reward mixes CNY units (`TP`, `DP`) with an inventory penalty
  whose scale depends on the `TRADE_UNIT` normalisation. Comparing reward values across
  configurations (the current improvement baseline uses `episode_reward`) is therefore
  meaningless, and it makes the reported `episode_reward` of 391.84 versus a PnL of 345.0
  hard to interpret.
* **Change:** report the reward decomposition per episode (already computed:
  `reward_dampened_pnl`, `reward_trading_pnl`, `reward_inventory_punishment`,
  `reward_spread_punishment`) and normalise the inventory penalty by a rolling estimate of the
  per-unit spread so that `zeta` is dimensionless.
* **Worked example:** at `Inv = 500` shares and `zeta = 0.01`, `IP = 0.25` per step, so a
  50-step episode at that inventory accumulates 12.5 of penalty against a measured episode PnL
  of 49.0 CNY. If instead `zeta` were applied to raw shares rather than lots,
  `IP = 0.01*500^2 = 2500` per step, which would swamp the PnL signal entirely. Both are the
  same formula with a different unit convention, which is exactly why the normalisation has to
  be explicit.
* **How to test:** assert `reward == r_ma*TP + r_da*DP - r_ip*IP` per step in a unit test, and
  that the sum of reward components matches `episode_reward`.
* **Effect size:** none on PnL, high on interpretability. **Confidence: high.**
  **Cost:** half a day.

### 13. Hyperparameter search with a fixed budget

* **Axis:** model and algorithm.
* **Hypothesis:** the paper states almost no hyperparameters, and the shipped values (lr 1e-4,
  batch 32, discount 0.99, memory 200000) are unsourced. They may be far from good for this
  problem, particularly the discount, which for per-step PnL at a 1 Hz book implies a horizon
  of about 100 steps, i.e. under two minutes.
* **Change:** Optuna over learning rate, entropy coefficient, discount, n-step horizon, and
  the reward coefficients `eta` and `zeta`, with early stopping on a validation day and the
  item 1 metrics as the objective.
* **Worked example (discount):** `0.99` at one decision per second gives a half-life of
  `ln(2)/ln(1/0.99) = 69` steps, so inventory decisions beyond about a minute are effectively
  discounted away. If the intended horizon is an episode of 2000 events, the discount should
  be nearer `0.999` (half-life 693 steps). The shipped value probably shortens the effective
  horizon to well below the episode.
* **How to test:** the search's held-out result versus the baseline, with the item 1 CI.
* **Effect size:** medium to large, and often the cheapest PnL gain. **Confidence: medium to
  high.** **Cost:** 2 to 4 days plus the compute for the sweep.

### 14. Reproduce the data pipeline from a legal source

* **Axis:** anything else, and the actual blocker for reproduction.
* **Hypothesis:** nothing else in this list can be compared against the paper's numbers without
  data that resembles the paper's. The repository cannot supply it (see section 6).
* **Change:** implement the LOB reconstruction from a level-2 order and trade feed to
  `ask/bid/price/msg`, either from a licensed CSMAR Shenzhen extract or from a substitute
  venue with a full order-by-order feed. The `convert_tardis` skeleton in `data/adapter.py`
  lists the mismatches: no order queue file, no cancellations, tick and lot sizes, session
  boundaries, and the absence of fees.
* **Worked example:** `msg.csv` needs market, limit and withdraw volumes and counts. A
  snapshot-plus-trades feed gives net level changes only, so a market buy of 500 shares and a
  cancellation of a 500 share resting bid are indistinguishable in the book, yet they map to
  opposite signs in the order strength index. Any reconstruction from snapshots therefore
  produces a systematically biased market state, which is why the paper's own pipeline is the
  only faithful source.
* **How to test:** verify that reconstructed book states match the exchange feed at every
  timestamp, and that the fill model's trade tape reproduces the exchange trade print sequence.
* **Effect size:** decisive for comparability. **Confidence: high.** **Cost:** 1 to 3 weeks,
  gated on data access.

### 15. Engineering hygiene: checkpoints, memory, reproducibility, dashboard

* **Axis:** anything else.
* **Hypothesis:** the current artifacts are large and the experiments are hard to compare.
* **Change:** stop persisting the replay buffer (N6), set `memory` explicitly instead of
  inheriting 66000 timesteps of 4 KB states, log a config hash and a git SHA into
  `results/{run_id}/config.json`, and add a small script that concatenates
  `test_daily.csv` across runs into one comparison table.
* **Worked example:** measured, `ckpt/agent/agent.npz` was 543 MB per save, of which 528 MB was
  `memory/states_lob_state-buffer.npy` at (66000, 50, 40, 1) float32. A 5-seed sweep would
  otherwise write about 2.7 GB of replay buffers.
* **How to test:** assert that a saved checkpoint is under a few MB and that a run can be
  reproduced from `config.json` alone.
* **Effect size:** none on PnL. **Confidence: high.** **Cost:** half a day.

### The first three I would do

1. **Item 1, evaluation rigor**, because every later item is unmeasurable without error bars
   and a PnL decomposition. It is cheap and it upgrades all subsequent work.
2. **Item 3, simulator realism, starting with fees**, because the verified PnL of 1.11 CNY per
   100-share unit is small enough that a plausible 0.7 CNY round-trip cost removes most of it.
   Any agent improvement measured before fees are modelled risks optimising a friction that
   does not exist.
3. **Item 2, baselines**, because an RL agent that does not beat a fixed-spread quoter after
   fees is not a contribution, and this is the cheapest way to find that out early.

Items 4 and 5 are the enablers that make the next round of experiments fast, and they should
follow immediately if more than a couple of experiments are planned.

---------------------------------------------------------------------------------------------------

## 6. What I could not verify

1. **Any paper number.** The dataset is not available, so nothing in this work is compared
   against the paper's reported results. Every PnL figure quoted here comes from the synthetic
   generator in `data/adapter.py` and is meaningless in absolute terms.
2. **The faithful TensorFlow 2.6 stack.** It cannot be installed on this machine (no arm64
   wheel) and the Dockerfile was not executed. The compat shim's no-op branch for TF 2.6 and
   older is exercised only by the feature probe in `compat.py`, not by a real TF 2.6 run.
   Everything verified here used tensorflow-macos 2.11.0 with the shim active.
3. **TensorForce version behaviour beyond 0.6.5.** The venv has 0.6.5 and the project's own
   `tensor` conda environment has 0.5.5, which has no Keras network support at all. I did not
   test intermediate or later versions.
4. **The paper's exact wording, first hand.** My own arXiv HTML fetch returned only about the first
   6 KB, so the citations in section 4 rest on a full read performed by a research sub-agent. The four
   items that drive code defaults (reward signs and the eta/zeta values, the 8 discrete actions and the
   flatten index, the FC-LOB sizes, the metric definitions) were re-read against the full text in a
   second pass and confirmed with verbatim quotations. One nuance is left to the reader: the paper
   says the FC-LOB list is (1024, 256, 64, 3) with Softmax "in the last layer" but never labels which
   entry is the classifier, so treating the trailing 3 as the 3-class head is inferred, not stated.
5. **Whether any of the shipped hyperparameters match the authors' actual runs.** Some conflict
   with the paper internally (for example `max_episode_timesteps=1000` against 2000-step
   episodes, and 8 train days against a described 10-day half), so at least part of the shipped
   code is not the code that produced the paper.
6. **The pretraining fidelity.** The volume normalisation mismatch (N3) is documented and
   unresolved, so a pretrained encoder from `pretrain.py` is not guaranteed to be aligned with
   the RL state in the way the authors' was.
7. **Real-market fill behaviour.** The queue model, partial fills and fees are unchanged from
   the shipped code. The measured fill behaviour comes from synthetic trades that print at the
   touch and one level deeper.
8. **The Dockerfile's `COPY . /workspace` happens after the dependency layers**, so a change to
   the source does not invalidate the pip cache; this is deliberate but it means the sanity
   check in the image runs against the code baked at build time, as usual.
9. **`pip check` cleanliness.** tensorforce 0.6.5 will always report unsatisfied metadata
   against this stack, which is documented in `requirements-macos.txt` rather than resolved.
10. **The `--load` path semantics.** It ran and produced PnL, but I did not verify that the
    restored agent is bit-identical to the saved one, only that it loads and trains.
