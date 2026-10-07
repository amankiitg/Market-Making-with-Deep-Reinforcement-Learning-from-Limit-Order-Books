from tensorforce.agents import Agent

from compat import apply_tensorforce_tf_compat

# TensorForce 0.6.5 touches private Keras optimizer internals in
# TFOptimizer.initialize_given_variables, which TensorFlow >= 2.11 removed. Installing the
# shim here guarantees it is active on every code path that creates an agent. It is a no-op
# on TensorFlow <= 2.10, see compat.py.
apply_tensorforce_tf_compat()

# `exploration` is a TensorForce parameter, either a float or a decay specification such as
# dict(type='linear', unit='updates', num_steps=2000, initial_value=1.0, final_value=0.01).
# TensorForce 0.6.5 defaults it to 0.0 (no exploration), which for a discrete DQN agent
# effectively freezes action selection at argmax of an untrained Q function. None is
# forwarded unchanged and means "use the TensorForce default".

def get_dueling_dqn_agent(
                        network, 
                        environment=None, 
                        states=None,
                        actions=None,
                        max_episode_timesteps=None,
                        batch_size=32, 
                        learning_rate=1e-4, 
                        horizon=1, 
                        discount=0.99,
                        memory=200000, 
                        device='gpu',
                        eager_mode=False,
                        exploration=None
                        ):
    if environment != None:
        agent = Agent.create(
        agent='dueling_dqn',
        environment=environment,
        max_episode_timesteps=max_episode_timesteps,
        network=network,
        config=dict(device=device, eager_mode=eager_mode),
        memory=memory,
        batch_size=batch_size, 
        learning_rate=learning_rate,
        horizon=horizon,
        discount=discount,
        parallel_interactions=10,
        exploration=exploration,
    )
    else:
        agent = Agent.create(
            agent='dueling_dqn',
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device, eager_mode=eager_mode),
            memory=memory,
            batch_size=batch_size, 
            learning_rate=learning_rate,
            horizon=horizon,
            discount=discount,
            parallel_interactions=10,
            exploration=exploration,
        )
    return agent

def get_ppo_agent(
                network, 
                environment=None, 
                states=None,
                actions=None,
                max_episode_timesteps=None,
                batch_size=32, 
                learning_rate=1e-3,
                horizon=None, 
                discount=0.99,
                device='gpu',
                eager_mode=False,
                exploration=None
                ):
    if environment != None:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device, eager_mode=eager_mode),
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            discount=discount,
            parallel_interactions=10,
            exploration=exploration,
        )
    else:
        agent = Agent.create(
            agent='ppo',
            environment=environment,
            states=states,
            actions=actions,
            max_episode_timesteps=max_episode_timesteps,
            network=network,
            config=dict(device=device, eager_mode=eager_mode),
            batch_size=batch_size, 
            learning_rate=learning_rate,
            discount=discount,
            parallel_interactions=10,
            exploration=exploration,
        )

    return agent
