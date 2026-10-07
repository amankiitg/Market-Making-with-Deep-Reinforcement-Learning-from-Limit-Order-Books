"""Single source of truth for the environment state layout.

The environments return a dict of tensors. TensorForce's ``KerasNetwork.apply`` calls the
Keras model with ``list(states.values())``, i.e. positional inputs in the *insertion order*
of the state dict. Therefore:

* ``EnvContinuous``/``EnvDiscrete`` must build ``states()`` (and the dict returned by
  ``get_state_at_t``) by iterating ``STATE_KEYS`` in order;
* ``network.get_model`` must build its Keras ``Input`` list in exactly the same order.

Both sides import this module so the two can never drift apart. ``tests/test_wiring.py``
enforces the contract (each declared state reaches the policy network exactly once, and
perturbing any single state changes the policy output).
"""

STATE_KEYS = ('lob_state', 'market_state', 'agent_state')

# Number of order book levels kept per side by the data loaders.
LOB_LEVELS = 10
# ask price, ask volume, bid price, bid volume per level.
LOB_CHANNELS = 4 * LOB_LEVELS


def _lob_columns():
    columns = []
    for level in range(1, LOB_LEVELS + 1):
        columns.extend((
            f'ask{level}_price', f'ask{level}_volume',
            f'bid{level}_price', f'bid{level}_volume',
        ))
    return tuple(columns)


# Canonical channel order of the flattened order book, price first per level, matching the
# paper's Eq. (1): {P^ask_i, V^ask_i, P^bid_i, V^bid_i} for i = 1..n. The repo's unused
# utils.reorder() documents a volume-first layout instead; the paper wins here, and both the
# RL state path (utils.lob_norm) and the pretraining path (pretrain.py) use this constant so
# the pretrained encoder and the RL encoder see identical channel semantics.
LOB_COLUMNS = _lob_columns()

# 6 factors (realised volatility and RSI over 300s/600s/1800s) + 18 order strength
# factors (6 order strength indices over 10s/60s/300s), see environment/env_feature.py.
MARKET_STATE_DIM = 24
# 12 inventory slots + 12 time slots, see Env*.get_state_at_t.
AGENT_STATE_DIM = 24


def active_state_keys(wo_lob_state=False, wo_market_state=False, wo_agent_state=False):
    """Return the canonical state keys minus the ablated ones, preserving order."""
    excluded = set()
    if wo_lob_state:
        excluded.add('lob_state')
    if wo_market_state:
        excluded.add('market_state')
    if wo_agent_state:
        excluded.add('agent_state')
    return tuple(key for key in STATE_KEYS if key not in excluded)


def check_canonical_order(keys):
    """Raise if ``keys`` is not a subsequence of ``STATE_KEYS``.

    Guards against callers reordering states, which would silently permute the network
    inputs (see the ordering note in network/network.py).
    """
    keys = tuple(keys)
    unknown = [key for key in keys if key not in STATE_KEYS]
    if unknown:
        raise ValueError(f'unknown state key(s): {unknown}; expected a subset of {STATE_KEYS}')
    expected = tuple(key for key in STATE_KEYS if key in set(keys))
    if keys != expected:
        raise ValueError(f'state keys {keys} are not in canonical order {expected}')
    if not keys:
        raise ValueError('at least one state key is required')
    return keys


def state_specs(time_window, keys=None, market_state_dim=MARKET_STATE_DIM,
                agent_state_dim=AGENT_STATE_DIM):
    """Build the TensorForce state specification dict for the given keys."""
    keys = STATE_KEYS if keys is None else check_canonical_order(keys)
    specs = {}
    for key in keys:
        if key == 'lob_state':
            specs[key] = dict(type='float', shape=(time_window, LOB_CHANNELS, 1))
        elif key == 'market_state':
            specs[key] = dict(type='float', shape=(market_state_dim,))
        elif key == 'agent_state':
            specs[key] = dict(type='float', shape=(agent_state_dim,))
        else:  # pragma: no cover - check_canonical_order already rejects this
            raise ValueError(f'unsupported state key {key!r}')
    return specs


def state_input_dims(time_window, keys=None, market_state_dim=MARKET_STATE_DIM,
                     agent_state_dim=AGENT_STATE_DIM):
    """Return ``{key: shape}`` (without the batch dimension) in canonical order."""
    keys = STATE_KEYS if keys is None else check_canonical_order(keys)
    dims = {}
    for key in keys:
        if key == 'lob_state':
            dims[key] = (time_window, LOB_CHANNELS, 1)
        elif key == 'market_state':
            dims[key] = (market_state_dim,)
        else:
            dims[key] = (agent_state_dim,)
    return dims
