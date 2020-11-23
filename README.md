<p align="center">
    <br>
    <img src="https://github.com/SforAiDl/genrl/blob/master/docs/source/assets/images/genrl.png" width="200"/>
    <br>
</p>
<div align='center'>

[![pypi](https://img.shields.io/badge/pypi%20package-v0.0.2-blue)](https://pypi.org/project/genrl/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/genrl.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Downloads](https://pepy.tech/badge/genrl)](https://pepy.tech/project/genrl)
[![codecov](https://codecov.io/gh/SforAiDl/genrl/branch/master/graph/badge.svg)](https://codecov.io/gh/SforAiDl/genrl)
[![GitHub license](https://img.shields.io/github/license/SforAiDl/genrl)](https://github.com/SforAiDl/genrl/blob/master/LICENSE)

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/SforAiDl/genrl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SforAiDl/genrl/context:python)
[![Maintainability](https://api.codeclimate.com/v1/badges/c3f6e7d31c078528e0e1/maintainability)](https://codeclimate.com/github/SforAiDl/genrl/maintainability)
[![CodeFactor](https://www.codefactor.io/repository/github/sforaidl/genrl/badge)](https://www.codefactor.io/repository/github/sforaidl/genrl)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/SforAiDl/genrl.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/SforAiDl/genrl/alerts/)

[![Build Status](https://travis-ci.com/SforAiDl/genrl.svg?branch=master)](https://travis-ci.com/SforAiDl/genrl)
[![Documentation Status](https://readthedocs.org/projects/genrl/badge/?version=latest)](https://genrl.readthedocs.io/en/latest/?badge=latest)
![Tests MacOS](https://github.com/SforAiDl/genrl/workflows/Tests%20MacOS/badge.svg)
![Tests Linux](https://github.com/SforAiDl/genrl/workflows/Tests%20Linux/badge.svg)
![Tests Windows](https://github.com/SforAiDl/genrl/workflows/Tests%20Windows/badge.svg)

[![Slack - Chat](https://img.shields.io/badge/Slack-Chat-blueviolet)](https://join.slack.com/t/genrlworkspace/shared_invite/zt-gwlgnymd-Pw3TYC~0XDLy6VQDml22zg)

</div>

[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/0)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/0)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/1)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/1)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/2)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/2)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/3)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/3)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/4)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/4)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/5)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/5)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/6)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/6)[![](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/images/7)](https://sourcerer.io/fame/Sharad24/Sharad24/genrl/links/7)

---

**GenRL is a PyTorch reinforcement learning library centered around reproducible, generalizable algorithm implementations and improving accessibility in Reinforcement Learning** 

**GenRL's current release is at v0.0.2. Expect breaking changes**

Reinforcement learning research is moving faster than ever before. In order to keep up with the growing trend and ensure that RL research remains reproducible, GenRL aims to aid faster paper reproduction and benchmarking by providing the following main features:

- **PyTorch-first**: Modular, Extensible and Idiomatic Python
- **Tutorials and Example**: 20+ Tutorials from basic RL to SOTA Deep RL algorithm (with explanations)!
- **Unified Trainer and Logging class**: code reusability and high-level UI
- **Ready-made algorithm implementations**: ready-made implementations of popular RL algorithms.
- **Faster Benchmarking**: automated hyperparameter tuning, environment implementations etc.

By integrating these features into GenRL, we aim to eventually support **any new algorithm implementation in less than 100 lines**.

**If you're interested in contributing, feel free to go through the issues and open PRs for code, docs, tests etc. In case of any questions, please check out the [Contributing Guidelines](CONTRIBUTING.md)**


## Installation

GenRL is compatible with Python 3.6 or later and also depends on `pytorch` and `openai-gym`. The easiest way to install GenRL is with pip, Python's preferred package installer.

    $ pip install genrl

Note that GenRL is an active project and routinely publishes new releases. In order to upgrade GenRL to the latest version, use pip as follows.

    $ pip install -U genrl

If you intend to install the latest unreleased version of the library (i.e from source), you can simply do:

    $ git clone https://github.com/SforAiDl/genrl.git
    $ cd genrl
    $ python setup.py install

## Usage
To train a Soft Actor-Critic model from scratch on the `Pendulum-v0` gym environment and log rewards on tensorboard
```python
import gym

from genrl.agents import SAC
from genrl.trainers import OffPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("Pendulum-v0")
agent = SAC('mlp', env)
trainer = OffPolicyTrainer(agent, env, log_mode=['stdout', 'tensorboard'])
trainer.train()
```

To train a Tabular Dyna-Q model from scratch on the `FrozenLake-v0` gym environment and plot rewards:
```python
import gym

from genrl.agents import QLearning
from genrl.trainers import ClassicalTrainer

env = gym.make("FrozenLake-v0")
agent = QLearning(env)
trainer = ClassicalTrainer(agent, env, mode="dyna", model="tabular", n_episodes=10000)
episode_rewards = trainer.train()
trainer.plot(episode_rewards)
```

## Tutorials
- [Multi Armed Bandits](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/bandit_overview.html)
    - [Upper Confidence Bound](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/ucb.html)
    - [Thompson Sampling](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/thompson_sampling.html)
    - [Bayesian](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/bayesian.html)
    - [Softmax Action Selection](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/gradients.html)
- [Contextual Bandits](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/contextual_overview.html)
    - [Linear Posterior Inference](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/linpos.html)
    - [Variational Inference](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/variational.html)
    - [https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/bootstrap.html](Bootstrap)
    - [Parameter Noise Sampling](https://genrl.readthedocs.io/en/latest/usage/tutorials/bandit/noise.html)
- [Deep Reinforcement Learning Background](https://genrl.readthedocs.io/en/latest/usage/tutorials/Deep/Background.html)
    - [Vanilla Policy Gradients](https://genrl.readthedocs.io/en/latest/usage/tutorials/Deep/VPG.html)
    - [Advantage Actor Critic](https://genrl.readthedocs.io/en/latest/usage/tutorials/Deep/A2C.html)
    - [Proximal Policy Optimization](https://genrl.readthedocs.io/en/latest/usage/tutorials/Deep/PPO.html)
    
## Algorithms

### Deep RL
 - DQN (Deep Q Networks)
    - DQN
    - Double DQN
    - Dueling DQN
    - Noisy DQN
    - Categorical DQN
 - VPG (Vanilla Policy Gradients)
 - A2C (Advantage Actor-Critic)
 - PPO (Proximal Policy Optimization)
 - DDPG (Deep Deterministic Policy Gradients)
 - TD3 (Twin Delayed DDPG)
 - SAC (Soft Actor Critic)

### Classical RL
 - SARSA
 - Q Learning

### Bandit RL
 - Multi Armed Bandits
    - Eps Greedy
    - UCB
    - Thompson Sampling
    - Bayesian Bandits
    - Softmax Explorer
  - Contextual Bandits
    - Eps Greedy
    - UCB
    - Thompson Sampling
    - Bayesian Bandits
    - Softmax Explorer
 - Deep Contextual Bandits
    - Variation Inference
    - Noise sampling for neural network parameters
    - Epsilon greedy with a neural network
    - Bayesian Regression on for posterior inference
    - Bootstraped Ensemble
    

#### Credits and Similar Libraries:
- [Gym](https://gym.openai.com/) - Environments 
- [Ray](https://github.com/ray-project/ray)
- [OpenAI Baselines](https://github.com/openai/baselines) - Logger
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3): Stable Baselines aims to provide _baselines_ for Deep RL Algorithms. 
- [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
- [Deep Contextual Bandits](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits)
