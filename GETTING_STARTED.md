## Getting Started

### Classical RL

#### Tabular Dyna-Q model:

Train a Tabular Dyna-Q model from scratch on the `FrozenLake-v0` gym environment and plot rewards

```python

env = gym.make("FrozenLake-v0")
agent = QLearning(env)
trainer = Trainer(agent, env, mode="dyna", model="tabular", n_episodes=10000)
episode_rewards = trainer.train()
trainer.plot(episode_rewards)
```

### Deep RL

#### Vanilla Policy Gradient

Train Vanilla Policy Gradient on Vectorized CartPole-v1

```python
from genrl import PPO1, SAC, VPG
from genrl.deep.common import OffPolicyTrainer, OnPolicyTrainer
from genrl.environments import VectorEnv

# Specify some hyperparameters
n_envs = 10
epochs = 15
eval_episodes = 10
arch = "mlp"
log = ["stdout"] # Specify logging type as a comma-separated list

# Initialize Agent and Environment
env = VectorEnv("CartPole-v1", n_envs)
agent = VPG(arch, env)

# Trainer
trainer = OnPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
trainer.train()

# Evaluation
trainer.evaluate(render=True)
```

#### Soft Actor-Critic (SAC)

Train Soft Actor-Critic (SAC) on Vectorized Pendulum-v0

```python


# Specify some hyperparameters
n_envs = 10
epochs = 40
eval_episodes = 20
arch = "mlp"
log = ["stdout", "csv"] # Specify logging type as a comma-separated list

# Initialize Agent and Environment
env = VectorEnv("Pendulum-v0", n_envs)
agent = SAC(arch, env)

# Trainer
trainer = OffPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
trainer.train()

# Evaluation
trainer.evaluate(render=True)
```


#### Proximal Policy Optimization (PPO)

Train Proximal Policy Optimization (PPO) on Vectorized Breakout-v0

```python


# Specify some hyperparameters
n_envs = 2
epochs = 100
eval_episodes = 20
arch = "cnn"
rollout_size = 128
log = ["stdout", "tensorboard"] # Specify logging type as a comma-separated list

# Initialize Agent and Environment
env = VectorEnv("Breakout-v0", n_envs, env_type = "atari")
agent = PPO1(arch, env, rollout_size = rollout_size)

# Trainer
trainer = OnPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
trainer.train()

# Evaluation
trainer.evaluate(render=True)
```
