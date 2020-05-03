# GenRL

**GenRL is a PyTorch reinforcement learning library centered around reproducible and generalizable algorithm implementations.** 

Reinforcement learning research is moving faster than ever before. In order to keep up with the growing trend and ensure that RL research remains reproducible, GenRL aims to aid faster paper reproduction and benchmarking by providing the following main features:

- **PyTorch-first**: Pythonic and modular
- **Unified Trainer and Logging class**: code reusability and high-level UI
- **Ready-made algorithm implementations**: ready-made implementations of popular RL algorithms.
- **Faster Benchmarking**: automated hyperparameter tuning, environment implementations etc.

By integrating these features into GenRL, we aim to eventually support **any new algorithm implementation in less than 100 lines**.

**If you're interested in contributing, feel free to go through the issues and open PRs for code, docs, tests etc. In case of any questions, please check out the [Contributing Guidelines](https://github.com/SforAiDl/genrl/wiki/Contributing-Guidelines)**


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
To train a Tabular Dyna-Q model from scratch on the `FrozenLake-v0` gym environment and plot rewards:
```python
from genrl.classical import Trainer
from genrl import QLearning
import gym

env = gym.make("FrozenLake-v0")
agent = QLearning(env)
trainer = Trainer(agent, env, mode="dyna", model="tabular", n_episodes=10000)
episode_rewards = trainer.train()
trainer.plot(episode_rewards)
```
