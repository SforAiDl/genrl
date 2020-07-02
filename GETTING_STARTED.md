## Getting Started

To train a Soft Actor-Critic model from scratch on the `CartPole-v0` gym environment and log rewards on tensorboard
```python
import gym

from genrl import SAC, QLearning
from genrl.classical.common import Trainer
from genrl.deep.common import OffPolicyTrainer

env = gym.make("CartPole-v0")
agent = SAC('mlp', env)
trainer = OffPolicyTrainer(agent, env, log_mode=['stdout', 'tensorboard'])
trainer.train()
```

To train a Tabular Dyna-Q model from scratch on the `FrozenLake-v0` gym environment and plot rewards:
```python

env = gym.make("FrozenLake-v0")
agent = QLearning(env)
trainer = Trainer(agent, env, mode="dyna", model="tabular", n_episodes=10000)
episode_rewards = trainer.train()
trainer.plot(episode_rewards)
```
