Example code to run the algo

```python
import gym

from genrl import A2C
from genrl.classical.common import Trainer
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("CartPole-v0")
agent = A2C(network_type='mlp', env=env, rollout_size=2048)
trainer = OnPolicyTrainer(agent, env, log_mode=['stdout', 'tensorboard'], log_key="Episode")
trainer.train()
```

Hyperparameters for the A2C agent 
(https://github.com/SforAiDl/genrl/blob/master/genrl/deep/agents/a2c/a2c.py): 

    network_type: Deep neural network layer types ("mlp")
    gamma: Discount factor
    lr_actor: Learning rate for the actor (policy update)
    lr_critic: Learning rate for the critic (value function update)
    batch_size: Batch size for updating parameters
    num_episodes: Number of episodes
    timesteps_per_actorbatch: Number of timesteps per epoch
    max_ep_len: Maximum timesteps in an episode
    layers: Number of neurons in hidden layers
    noise: Noise function to use
    noise_std: Standard deviation for action noise
    seed: Seed for reproducing results
    render: True if environment is to be rendered, else False
    rollout_size: Timesteps for which the agent runs before updating parameters (Length of 1 epoch)
    val_coeff: Coefficient of value loss in overall loss function
    entropy_coeff: Coefficient of entropy loss in overall loss function

Hyperparameters for the OnPolicyTrainer
(https://github.com/SforAiDl/genrl/blob/master/genrl/deep/common/trainer.py):

    log_key: Key to be plotted on the x_axis
    save_interval: Model to save in each of these many timesteps
    max_ep_len: Max Episode Length
    steps_per_epochs: Steps to take per epoch
    epochs: Total Epochs to train for
    log_interval: Log important params every these many steps
