import os
import inspect
import json
import random
import time
from typing import List

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import pyrallis
from dataclasses import asdict, dataclass, field

from compat import apply_tensorforce_tf_compat
from state_spec import active_state_keys

from environment.env_discrete import EnvDiscrete
from environment.env_continuous import EnvContinuous
from agent.tensorforce_agent import get_dueling_dqn_agent, get_ppo_agent
from network.network import (LOB_LATENT_DIM, get_fclob_model, get_lob_model, get_model,
                             get_pretrain_model, make_compute_output_shape)

# TensorForce 0.6.5 calls private Keras optimizer internals that TensorFlow >= 2.11 removed.
# No-op on older TensorFlow, see compat.py.
apply_tensorforce_tf_compat()

# Per-episode result columns, written both to the episode tables and to ./results/{run_id}.
RESULT_COLUMNS = [
    'PnL', 'ND-PnL', 'PnL-MAP', 'profit_ratio', 'average_position', 'average_abs_position',
    'average_spread', 'volume', 'traded_units', 'holding_pnl', 'trading_pnl', 'episode_reward'
]


@dataclass
class TrainConfig:
    # Experiment
    code: str = '000001'
    device: str = "cpu"
    latency: int = 1
    time_window: int = 50
    log: bool = False
    exp_name: str = ''
    seed: int = 0
    # Data and episodes. Paper section IV-B1: 21 Shenzhen trading days in November 2019,
    # first half for training, second half for testing. The shipped date lists are shorter
    # (8 train / 13 test days) and do not match the paper exactly.
    train_days: List[str] = field(default_factory=lambda: [
                  '20191101', '20191104', '20191105', '20191106', '20191107', '20191108',
                  '20191111', '20191112'])
    test_days: List[str] = field(default_factory=lambda: [
                  '20191113', '20191114', '20191115', '20191118', '20191119', '20191120',
                  '20191121', '20191122', '20191125', '20191126', '20191127', '20191128',
                  '20191129'])
    num_step_per_episode: int = 2000
    n_train_loop: int = 5
    # Agent
    agent_type: str = 'ppo' # ppo/dueling dqn
    learning_rate: float = 1e-4
    horizon: int = 1
    env_type: str = 'continuous' # continuous/discrete
    max_episode_timesteps: int = 2000
    # TensorForce defaults exploration to 0.0, i.e. no exploration, which for the
    # discrete dueling DQN leaves action selection greedy from the first update. Set a
    # positive value, or pass a decay spec on the command line, to explore.
    exploration: float = 0.0
    # Agent.act defaults deterministic=True, and TensorForce skips exploration entirely when
    # deterministic is true (core/models/tensorforce.py: tf.cond on deterministic OR
    # exploration == 0). The shipped loop called act(independent=True) with that default, so
    # it trained on the greedy action and never explored. Sample while training, act greedily
    # when testing; set both to True to recover the original behaviour.
    train_deterministic: bool = False
    test_deterministic: bool = True
    load: bool = False
    agent_load_dir: str = ''
    save: bool = False
    agent_save_dir: str = './ckpt/agent'
    agent_save_filename: str = 'agent'
    keras_model_dir: str = './ckpt/agent_model'
    # Reward. Paper section III-D Eq. (16) uses the composite reward, so that is the default
    # here; 'pnl' reproduces the shipped behaviour of the original repository.
    reward_mode: str = 'composite'
    inventory_penalty: float = 0.01   # zeta, paper section IV-C2
    asymmetry_eta: float = 0.5        # eta, paper section IV-C2
    spread_penalty_lambda: float = 100.0  # non-paper, only used with --reward_mode=pnl
    # Actions. Paper section III-C1 defines 8 discrete actions (0..7, 7 flattens the book).
    discrete_num_actions: int = 8
    max_bias: float = 0.05            # paper section IV-C2
    max_spread: float = 0.1           # paper section IV-C2
    # Ablation
    wo_pretrain: bool = False
    wo_attnlob: bool = False
    wo_lob_state: bool = False
    wo_market_state: bool = False
    wo_agent_state: bool = False
    wo_dampened_pnl: bool = False
    wo_matched_pnl: bool = False
    wo_inv_punish: bool = False
    # Output
    results_dir: str = './results'
    run_id: str = ''


def init_env(day, config):
    kwargs = dict(
        code=config['code'],
        day=day,
        latency=config['latency'],
        T=config['time_window'],
        # state ablation
        wo_lob_state=config['wo_lob_state'],
        wo_market_state=config['wo_market_state'],
        wo_agent_state=config['wo_agent_state'],
        # reward ablation
        wo_dampened_pnl=config['wo_dampened_pnl'],
        wo_matched_pnl=config['wo_matched_pnl'],
        wo_inv_punish=config['wo_inv_punish'],
        # reward definition
        reward_mode=config['reward_mode'],
        inventory_penalty=config['inventory_penalty'],
        asymmetry_eta=config['asymmetry_eta'],
        # exp setting
        experiment_name=config['exp_name'],
        log=config['log'],
        )

    if config['env_type'] == 'continuous':
        environment = EnvContinuous(
            max_bias=config['max_bias'],
            max_spread=config['max_spread'],
            spread_penalty_lambda=config['spread_penalty_lambda'],
            **kwargs)
    elif config['env_type'] == 'discrete':
        environment = EnvDiscrete(num_actions=config['discrete_num_actions'], **kwargs)
    else:
        raise ValueError(f"env_type must be 'continuous' or 'discrete', got {config['env_type']!r}")
    return environment


def build_lob_encoder(config):
    """Build the (optionally pretrained) LOB encoder shared by all policy heads."""
    if config['wo_attnlob']:
        print("Ablation: attnlob")
        return get_fclob_model(LOB_LATENT_DIM, config['time_window'])

    model = get_lob_model(LOB_LATENT_DIM, config['time_window'])
    # Keras cannot infer the MultiHeadAttention output shape, see network.make_compute_output_shape.
    model.compute_output_shape = make_compute_output_shape(LOB_LATENT_DIM)

    if config['wo_pretrain']:
        print("Ablation: pretrain")
        return model

    pretrain_model_dir = f'./ckpt/pretrain_model_' + config['code']
    checkpoint_filepath = pretrain_model_dir + '/weights'
    if not os.path.exists(checkpoint_filepath + '.index'):
        raise FileNotFoundError(
            f'Pretrained LOB encoder not found at {checkpoint_filepath}.\n'
            f'Train it first with `python pretrain.py --code {config["code"]}`, or run the '
            f'agent with --wo_pretrain=True to skip the pretrained encoder.'
        )
    model_pretrain = get_pretrain_model(model, config['time_window'])
    model_pretrain.load_weights(checkpoint_filepath)
    return model_pretrain.layers[1]


def init_agent(environment, config):
    kwargs=dict()
    if config['agent_type'] == 'dueling_dqn':
        get_agent = get_dueling_dqn_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']
    elif config['agent_type'] == 'ppo':
        get_agent = get_ppo_agent
        kwargs['learning_rate']=config['learning_rate']
        kwargs['horizon']=config['horizon']
    else:
        raise ValueError(f"agent_type must be 'ppo' or 'dueling_dqn', got {config['agent_type']!r}")

    state_keys = active_state_keys(
        wo_lob_state=config['wo_lob_state'],
        wo_market_state=config['wo_market_state'],
        wo_agent_state=config['wo_agent_state'],
    )
    lob_model = build_lob_encoder(config)
    model = get_model(lob_model, config['time_window'], state_keys=state_keys)
    if config['exploration']:
        kwargs['exploration'] = config['exploration']
    agent = get_agent(
        model,
        environment=environment,
        max_episode_timesteps=config['max_episode_timesteps'],
        device=config['device'],
        **kwargs)

    if config['load']:
        model = keras.models.load_model(config['keras_model_dir'])
        model.layers[1].compute_output_shape = make_compute_output_shape(LOB_LATENT_DIM)
        agent = get_agent(
            model,
            environment=environment,
            max_episode_timesteps=config['max_episode_timesteps'],
            device=config['device'],
            **kwargs)
        agent.restore(
            config['agent_load_dir'],
            filename=config['agent_save_filename'],
            format='numpy')

    return agent

def train_a_day(environment, agent, train_result, config):
    num_step_per_episode = config['num_step_per_episode']
    num_episodes = len(environment.orderbook)//num_step_per_episode
    for idx in tqdm(range(num_episodes)):
        episode_states = list()
        episode_actions = list()
        episode_terminal = list()
        episode_reward = list()

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        if states is None:
            # The window has too few trade ticks; reset_seq refuses it instead of raising
            # StopIteration, so there is nothing to learn from.
            continue
        terminal = False
        while not terminal:
            episode_states.append(states)
            actions = agent.act(states=states, independent=True,
                                deterministic=config['train_deterministic'])
            episode_actions.append(actions)
            states, terminal, reward = environment.execute(actions=actions)
            episode_terminal.append(terminal)
            episode_reward.append(reward)

        agent.experience(
            states=episode_states, 
            actions=episode_actions, 
            terminal=episode_terminal,
            reward=episode_reward
        )

        agent.update()
        
        save_episode_result(environment, train_result)

    return episode_states, episode_actions, episode_reward

def test_a_day(environment, agent, test_result, config):
    num_step_per_episode = config['num_step_per_episode']
    num_episodes = len(environment.orderbook)//num_step_per_episode
    for idx in tqdm(range(num_episodes)):

        states = environment.reset_seq(timesteps_per_episode=num_step_per_episode, episode_idx=idx)
        if states is None:
            continue
        terminal = False
        while not terminal:
            actions = agent.act(
                states=states, independent=True,
                deterministic=config['test_deterministic']
            )
            states, terminal, reward = environment.execute(actions=actions)
        
        save_episode_result(environment, test_result)

def train(agent, train_result, config):
    for day in config['train_days']:
        environment = init_env(day, config)
        train_a_day(environment, agent, train_result, config)

def test(agent, test_result, config):
    for day in config['test_days']:
        environment = init_env(day, config)
        test_a_day(environment, agent, test_result, config)

def save_episode_result(environment, test_result):
    res_dict = environment.get_final_result()
    date = environment.day
    idx = environment.episode_idx

    test_result.loc[date+'_'+str(idx)] = [
        res_dict['pnl'], res_dict['nd_pnl'], res_dict['pnl_map'], res_dict['profit_ratio'],
        res_dict['avg_position'], res_dict['avg_abs_position'], res_dict['avg_spread'],
        res_dict['volume'], res_dict['traded_units'], res_dict['holding_pnl'],
        res_dict['trading_pnl'], res_dict['episode_reward']]

def gather_test_results(test_result):
    day_list = list(test_result.index)
    for i in range(len(day_list)):
        day_list[i] = day_list[i][:10]
    day_list = set(day_list)
    gathered_results = pd.DataFrame(columns=RESULT_COLUMNS, dtype=float)
    for day in day_list:
        result = test_result[test_result.index.str.contains(day)]
        pnl = result.PnL.sum()
        nd_pnl = result['ND-PnL'].sum()
        ap = result.average_position.mean()
        # Total trading volume is summed over episodes. The original inverted it from
        # profit_ratio, which only worked while profit_ratio was value/volume.
        volume = result.volume.sum()
        pr = (pnl/volume) if volume else np.nan
        gathered_results.loc[day] = [
            pnl, nd_pnl, result['PnL-MAP'].mean(), pr, ap,
            result.average_abs_position.mean(), result.average_spread.mean(),
            volume, result.traded_units.sum(), result.holding_pnl.sum(),
            result.trading_pnl.sum(), result.episode_reward.sum()]
    gathered_results=gathered_results.sort_index()
    return gathered_results

def save_run_results(config, run_dir, train_result, test_result, daily_test_results):
    """Persist the config snapshot and the episode/daily tables, see README."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=2, default=str)
    train_result.to_csv(os.path.join(run_dir, 'train_episodes.csv'))
    test_result.to_csv(os.path.join(run_dir, 'test_episodes.csv'))
    daily_test_results.to_csv(os.path.join(run_dir, 'test_daily.csv'))
    print('Saved run outputs to', run_dir)


def save_keras_model(model, path):
    """Save a Keras model, passing save_format only where the API still accepts it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if 'save_format' in inspect.signature(model.save).parameters:
        model.save(path, save_format='tf')
    else:
        model.save(path)


def save_agent(agent, config):
    # save agent network
    save_keras_model(agent.model.policy.network.keras_model, config['keras_model_dir'])
    # Save agent
    os.makedirs(config['agent_save_dir'], exist_ok=True)
    agent.save(
        config['agent_save_dir'],
        filename=config['agent_save_filename'],
        format='numpy')

@pyrallis.wrap()
def main(config: TrainConfig):
    config = asdict(config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    run_id = config['run_id'] or time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['results_dir'], run_id)
    print('Run id:', run_id)

    environment = init_env(config['train_days'][0], config)
    agent = init_agent(environment, config)

    train_result = pd.DataFrame(columns=RESULT_COLUMNS, dtype=float)
    for _ in range(config['n_train_loop']):
        train(agent, train_result, config)
        if config['save']:
            save_agent(agent, config)

    test_result = pd.DataFrame(columns=RESULT_COLUMNS, dtype=float)
    test(agent, test_result, config)
    daily_test_results = gather_test_results(test_result)

    save_run_results(config, run_dir, train_result, test_result, daily_test_results)
    print(daily_test_results)

if __name__ == '__main__':
    main()
