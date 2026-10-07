"""State-wiring tests.

These lock down the contract described in state_spec.py: the environment builds its state
dict by iterating `STATE_KEYS`, TensorForce feeds the Keras model with
`list(states.values())` (tensorforce/core/networks/keras.py, `KerasNetwork.apply`), and
`network.get_model` must therefore create its Keras inputs in that same order, with every
input reaching the concatenation layer exactly once.

Three independent checks:
  (i)   structural: every Keras Input is an ancestor of exactly one Concatenate input;
  (ii)  sensitivity: perturbing any positional input changes the model output;
  (iii) end-to-end: TensorForce is hooked and does receive the state blocks in canonical
        order, verified with distinguishable sentinel values.
"""

import numpy as np
import pytest

pytest.importorskip('tensorflow')
pytest.importorskip('tensorforce')

from tensorflow import keras

from compat import apply_tensorforce_tf_compat
from network.network import LOB_LATENT_DIM, get_lob_model, get_model, make_compute_output_shape
from state_spec import AGENT_STATE_DIM, MARKET_STATE_DIM, active_state_keys

apply_tensorforce_tf_compat(verbose=False)

TIME_WINDOW = 50


def _flatten_tensors(x):
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten_tensors(item))
        return out
    return [x]


def _input_ancestors(tensor):
    """Names of the InputLayer-origin tensors that `tensor` depends on."""
    found = set()
    seen = set()
    stack = [tensor]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        history = getattr(current, '_keras_history', None)
        if history is None:
            continue
        layer, node_index = history[0], history[1]
        if layer.__class__.__name__ == 'InputLayer':
            found.add(layer.name)
            continue
        node = layer.inbound_nodes[node_index]
        stack.extend(_flatten_tensors(node.input_tensors))
    return found


def build_policy_model(state_keys, latent_dim=LOB_LATENT_DIM, time_window=TIME_WINDOW):
    encoder = get_lob_model(latent_dim, time_window)
    encoder.compute_output_shape = make_compute_output_shape(latent_dim)
    return get_model(encoder, time_window, state_keys=state_keys)


def test_keras_inputs_follow_canonical_state_order():
    state_keys = active_state_keys()
    model = build_policy_model(state_keys)
    assert len(model.inputs) == len(state_keys)
    assert [tensor.name.split(':')[0] for tensor in model.inputs] == [
        key for key in state_keys]


def test_every_state_reaches_concat_exactly_once():
    state_keys = active_state_keys()
    model = build_policy_model(state_keys)

    concat_layers = [layer for layer in model.layers
                     if isinstance(layer, keras.layers.Concatenate)]
    assert len(concat_layers) == 1, 'expected exactly one concatenation layer'
    slots = [_input_ancestors(tensor)
             for tensor in _flatten_tensors(concat_layers[0].input)]

    # one slot per state, and each declared state feeding exactly one slot
    assert len(slots) == len(state_keys)
    covered = set()
    for index, slot in enumerate(slots):
        assert len(slot) == 1, (
            f'concat slot {index} depends on {sorted(slot)}; expected exactly one state. '
            f'This is the duplicate-append bug shape (agent_state fed into the '
            f'market_state branch).'
        )
        covered |= slot
    assert covered == set(tensor.name for tensor in model.inputs)

    slot_by_state = {next(iter(slot)): index for index, slot in enumerate(slots)}
    for state_key in state_keys:
        assert state_key in slot_by_state, f'{state_key} never reaches the concat layer'
    assert [slot_by_state[key] for key in state_keys] == list(range(len(state_keys)))


def _model_input_shapes(model):
    """Per-input shapes without the batch dimension, in positional (feed) order."""
    shapes = model.input_shape
    if not isinstance(shapes, list):
        shapes = [shapes]
    return [tuple(shape)[1:] for shape in shapes]


def test_perturbing_each_state_changes_the_output():
    state_keys = active_state_keys()
    model = build_policy_model(state_keys)

    rng = np.random.default_rng(0)
    shapes = _model_input_shapes(model)
    assert len(shapes) == len(state_keys)
    inputs = [rng.random((2,) + shape).astype('float32') for shape in shapes]

    base = model(inputs, training=False).numpy()
    for index, key in enumerate(state_keys):
        perturbed = [array.copy() for array in inputs]
        perturbed[index] = perturbed[index] + 1.0
        changed = model(perturbed, training=False).numpy()
        assert not np.allclose(base, changed), (
            f'perturbing {key} (positional input {index}) does not change the output, so it '
            f'is not wired into the network.'
        )


SENTINEL_MARKET = 99.0
SENTINEL_AGENT = -7.0


def _make_sentinel_env(dataset):
    """EnvContinuous whose market/agent blocks are constant, distinguishable sentinels."""
    from environment.env_continuous import EnvContinuous

    class SentinelEnv(EnvContinuous):
        def get_state_at_t(self, t):
            state = super().get_state_at_t(t)
            state['market_state'] = [SENTINEL_MARKET] * MARKET_STATE_DIM
            state['agent_state'] = [SENTINEL_AGENT] * AGENT_STATE_DIM
            return state

    return SentinelEnv(
        code=dataset['code'],
        day=dataset['day'],
        latency=1,
        T=TIME_WINDOW,
        log=0,
        experiment_name='pytest-sentinel',
        data_dir=dataset['data_root'],
        raw_dir=dataset['raw_root'],
    )


def test_tensorforce_feeds_states_in_canonical_order(dataset):
    from tensorforce.core.networks.keras import KerasNetwork

    from agent.tensorforce_agent import get_ppo_agent

    env = _make_sentinel_env(dataset)
    state_keys = active_state_keys()
    model = build_policy_model(state_keys)
    # eager_mode executes the TensorForce tf.function bodies eagerly, so the recorded
    # tensors are concrete values rather than graph placeholders.
    agent = get_ppo_agent(model, environment=env, max_episode_timesteps=100,
                          device='cpu', learning_rate=1e-4, horizon=1, eager_mode=True)

    recorded = []
    original_apply = KerasNetwork.apply

    def spy(self, *args, **kwargs):
        x = kwargs.get('x', args[0] if args else None)
        try:
            recorded.append([np.array(value) for value in x.values()])
        except Exception:  # pragma: no cover - only if the hook sees a non-eager tensor
            recorded.append(None)
        return original_apply(self, *args, **kwargs)

    KerasNetwork.apply = spy
    try:
        state = env.reset_seq(timesteps_per_episode=100, episode_idx=0)
        assert state is not None
        agent.act(states=state, independent=True)
    finally:
        KerasNetwork.apply = original_apply

    records = [record for record in recorded if record is not None]
    assert records, 'KerasNetwork.apply was never called, the probe observed nothing'
    positional_inputs = records[-1]
    assert len(positional_inputs) == len(state_keys), (
        f'TensorForce fed {len(positional_inputs)} inputs for {len(state_keys)} states'
    )

    observed = []
    for values in positional_inputs:
        if np.allclose(values, SENTINEL_MARKET):
            observed.append('market_state')
        elif np.allclose(values, SENTINEL_AGENT):
            observed.append('agent_state')
        else:
            observed.append('lob_state')
    assert observed == list(state_keys), (
        f'TensorForce fed the states positionally as {observed}, but the Keras model was '
        f'built for {list(state_keys)}'
    )
