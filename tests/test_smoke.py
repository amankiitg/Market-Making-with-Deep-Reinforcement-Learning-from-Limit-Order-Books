"""Smoke tests: the environment, reward modes and configuration behave as documented."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip('tensorflow')
pytest.importorskip('tensorforce')

from conftest import make_env
from state_spec import STATE_KEYS, active_state_keys


def run_episode(env, rng, steps=200, action=None):
    """Drive one episode with the environment's own iterate/execute API."""
    state = env.reset_seq(timesteps_per_episode=steps, episode_idx=0)
    assert state is not None, 'fixture dataset must contain a usable episode'
    rewards = []
    for _ in range(10 ** 4):
        if action is None:
            if isinstance(env.actions(), dict) and env.actions().get('type') == 'int':
                chosen = int(rng.integers(0, env.actions()['num_values']))
            else:
                chosen = rng.random(2)
        else:
            chosen = action
        state, terminal, reward = env.execute(actions=chosen)
        rewards.append(reward)
        if terminal:
            break
    else:  # pragma: no cover
        pytest.fail('episode never terminated')
    return rewards


def test_continuous_episode_and_pnl_identity(env_continuous):
    env = env_continuous
    rewards = run_episode(env, np.random.default_rng(0))
    assert len(rewards) > 10

    result = env.get_final_result()
    # value = cash + inventory * mid, so the episode PnL must split exactly into the
    # execution (trading) and mark-to-market (holding) components.
    assert result['pnl'] == pytest.approx(result['holding_pnl'] + result['trading_pnl'],
                                          abs=1e-6)
    assert result['volume'] > 0
    assert result['traded_units'] > 0
    # ratios are either finite or nan (no inf, no ZeroDivisionError)
    for key in ('nd_pnl', 'pnl_map', 'profit_ratio'):
        assert np.isnan(result[key]) or np.isfinite(result[key]), (key, result[key])


def test_continuous_action_space_matches_paper(env_continuous):
    actions = env_continuous.actions()
    assert actions['shape'] == (2,)
    # paper section III-C2, Eq. (8)-(10): A1, A2 in [0, 1]
    assert actions['min_value'] == 0.0
    assert actions['max_value'] == 1.0


def test_states_follow_canonical_keys(env_continuous):
    assert tuple(env_continuous.states().keys()) == STATE_KEYS
    assert env_continuous.state_keys == STATE_KEYS
    state = env_continuous.reset_seq(timesteps_per_episode=100, episode_idx=0)
    assert tuple(state.keys()) == STATE_KEYS
    assert np.asarray(state['lob_state']).shape == (50, 40, 1)
    assert len(state['market_state']) == 24
    assert len(state['agent_state']) == 24


def test_state_ablation_removes_the_state_everywhere(dataset):
    env = make_env(dataset, wo_market_state=True)
    assert env.state_keys == active_state_keys(wo_market_state=True)
    assert 'market_state' not in env.states()
    state = env.reset_seq(timesteps_per_episode=100, episode_idx=0)
    assert 'market_state' not in state


def test_reward_modes_and_ablations_are_effective(dataset):
    """The wo_* flags must actually change the reward under reward_mode='composite'."""
    def episode_reward(overrides):
        np.random.seed(0)
        env = make_env(dataset, **overrides)
        rewards = run_episode(env, np.random.default_rng(0))
        return rewards, env.get_final_result()

    baseline, baseline_result = episode_reward({'reward_mode': 'composite'})
    ablated, _ = episode_reward({'reward_mode': 'composite', 'wo_matched_pnl': True})
    plain_pnl, _ = episode_reward({'reward_mode': 'pnl'})

    assert baseline != ablated, 'wo_matched_pnl must change the reward'
    assert baseline != plain_pnl, 'reward_mode must change the reward'
    # the PnL itself is a property of the fills, not of the reward shaping
    assert baseline_result['pnl'] == pytest.approx(
        episode_reward({'reward_mode': 'composite', 'wo_matched_pnl': True})[1]['pnl'])


def test_pnl_mode_without_spread_penalty_is_plain_pnl(dataset):
    np.random.seed(0)
    env = make_env(dataset, reward_mode='pnl', spread_penalty_lambda=0.0)
    rewards = run_episode(env, np.random.default_rng(0))
    result = env.get_final_result()
    # reward == sum of per-step value changes == episode PnL when no shaping is applied
    assert sum(rewards) == pytest.approx(result['pnl'], abs=1e-6)


def test_discrete_action_space_configurable(env_discrete):
    # paper section III-C1: 8 actions, action 7 flattens the position
    assert env_discrete.actions()['num_values'] == 8
    assert env_discrete.num_actions == 8


def test_discrete_every_configured_action_is_implementable(env_discrete):
    env = env_discrete
    env.reset_seq(timesteps_per_episode=100, episode_idx=0)
    for action in range(env.num_actions):
        orders = env.action2order(action)
        assert set(orders) == {'ask_price', 'ask_vol', 'bid_price', 'bid_vol'}
        assert orders['ask_vol'] <= 0 <= orders['bid_vol']


def test_discrete_agent_emits_every_configured_action(dataset):
    from agent.tensorforce_agent import get_dueling_dqn_agent
    from network.network import LOB_LATENT_DIM, get_lob_model, get_model, make_compute_output_shape
    from state_spec import active_state_keys

    np.random.seed(0)
    env = make_env(dataset, env_type='discrete')
    state_keys = active_state_keys()
    encoder = get_lob_model(LOB_LATENT_DIM, env.T)
    encoder.compute_output_shape = make_compute_output_shape(LOB_LATENT_DIM)
    model = get_model(encoder, env.T, state_keys=state_keys)
    # exploration=1.0 selects actions uniformly. Two things are needed for that to take
    # effect: TensorForce 0.6.5 defaults exploration to 0.0, and it skips exploration
    # entirely when `deterministic` is true, which is Agent.act's default in independent
    # mode (core/models/tensorforce.py, tf.cond on deterministic). See the report.
    agent = get_dueling_dqn_agent(model, environment=env, max_episode_timesteps=200,
                                 device='cpu', learning_rate=1e-4, horizon=1, exploration=1.0)

    state = env.reset_seq(timesteps_per_episode=200, episode_idx=0)
    seen = set()
    for _ in range(400):
        action = agent.act(states=state, independent=True, deterministic=False)
        seen.add(int(action))
        state, terminal, reward = env.execute(actions=action)
        if terminal:
            break
    assert seen <= set(range(env.num_actions)), f'agent emitted out-of-range action {seen}'
    assert seen == set(range(env.num_actions)), (
        f'agent did not explore every configured action, saw {sorted(seen)}'
    )


def test_reset_seq_refuses_unusable_window(sparse_dataset):
    env = make_env(sparse_dataset)
    assert env.reset_seq(timesteps_per_episode=100, episode_idx=0) is None


def test_train_config_defaults():
    from main import TrainConfig

    config = TrainConfig()
    # regression: a trailing comma once made this the truthy tuple (False,)
    assert config.save is False
    fields = {field.name for field in dataclasses.fields(TrainConfig)}
    for name in ('train_days', 'test_days', 'num_step_per_episode', 'n_train_loop',
                 'max_episode_timesteps', 'keras_model_dir', 'agent_save_filename',
                 'reward_mode', 'discrete_num_actions', 'results_dir'):
        assert name in fields, f'{name} must be a configuration field, not a module global'


def test_pretrain_dataset_shapes(dataset):
    from pretrain import PretrainConfig, build_dataset

    config = dataclasses.asdict(PretrainConfig(
        code=dataset['code'], days=[dataset['day']], data_dir=dataset['data_root'],
        time_window=50, horizon=10))
    X, Y = build_dataset(config)
    assert X.ndim == 4 and X.shape[1:] == (50, 40, 1)
    assert Y.shape == (X.shape[0], 3)


def test_adapter_validate_dataset(dataset):
    from data.adapter import validate_dataset

    validate_dataset(dataset['data_root'], dataset['raw_root'], dataset['code'],
                     dataset['day'])
