# Market Making with Deep RL from Limit Order Books (archived fork)

**Archived.** This reproduction cannot be completed because the paper's data was never released.
The follow-on project is [mm-rl-vs-optimum](https://github.com/amankiitg/mm-rl-vs-optimum):
PPO graded against the exact optimal market-making policy, in an equity and a corporate-bond RFQ setting.

## What this was

A fork of [imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books](https://github.com/imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books),
the demo code for "Market Making with Deep Reinforcement Learning from Limit Order Books"
(IJCNN 2023, [arXiv:2305.15821](https://arxiv.org/abs/2305.15821)).

## Why it cannot be reproduced

- **Data:** the paper uses licensed Shenzhen level-2 order and trade data (CSMAR) for November 2019. It was never published.
- **Pretrained encoder:** the checkpoint and the script that trains it were never published.

## What this fork fixed

The upstream code could not start. This fork makes it run end to end on synthetic data:

- **Fatal bugs:** a truthy `save` default, an undefined `keras_model_dir`, and a pandas 2 incompatibility.
- **State wiring:** the policy network consumed the agent state twice and never saw the market state.
- **Exploration:** the training loop acted deterministically, so TensorForce never explored.
- **Paper fidelity:** the paper's reward and its 8-action discrete space were unreachable as shipped.
- **Missing pieces:** added a pretraining script, a synthetic data generator, an Apple Silicon install path, and tests.

Details are in `REPORT.md` and the commit history.
