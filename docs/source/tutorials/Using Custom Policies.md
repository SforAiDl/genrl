### Custom Policy networks

GenRL provides custom policies for images (CNNPolicy) and for other types of inputs(MlpPolicy).  
One way to customize this policy is to create a class deriving from these classes, eg: 
```python
# Define your custom Policy
import torch
import torch.nn as nn

from genrl import VPG
from genrl.deep.common import (
    BasePolicy,
    BasePolicyPolicy,
    MlpPolicy,
    OnPolicyTrainer,
    mlp,
)
from genrl.environments import VectorEnv


# Define a custom MLP Policy
class custom_policy(MlpPolicy):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_policy, self).__init__(self, state_dim, 
                                            action_dim, 
                                            kwargs.get("hidden"))
        self.state_dim = state_dim
        self.action_dim = action_dim


# Initialize an environment
env = VectorEnv("CartPole-v0", 1)

# Initialize the custom Policy
state_dim = env.observation_space[0]
action_dim = env.action_space.n
policy = custom_policy(state_dim = state_dim, action_dim = action_dim,
                        hidden = (64, 64))

algo = VPG(custom_policy, env)

# Initialize the trainer and start training 
trainer = OnPolicyTrainer(algo, env, log_mode=["csv"],
                         logdir="./logs", epochs=100)
trainer.train()
```

You can also redefine the policy directly if you require more control.  
Say you want to use an LSTM followed by MLP layers for the policy.

```python




class custom_policy(BasePolicy):
    def __init__(self, state_dim, hidden, action_dim,
                 discrete = True, layer_size= 512, layers = 1, **kwargs):
        super(custom_policy, self).__init__(state_dim,
                                            action_dim,
                                            hidden,
                                            discrete,
                                            **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.lstm = nn.LSTM(self.state_dim, layer_size, layers)
        self.fc = mlp([layer_size] + list(hidden) + [action_dim],
                      sac=self.sac)  # the mlp layers

    def forward(self, state):
        state, h = self.lstm(state.unsqueeze(0))
        state = state.view(-1, self.layer_size)
        action = self.fc(state)
        return action


# Initialize an environment
env = VectorEnv("CartPole-v0", 1)

# Initialize the custom Policy
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = custom_policy(state_dim=state_dim, action_dim=action_dim,
                        hidden = (64, 64))

algo = VPG(policy, env)

# Initialize the trainer and start training 
trainer = OnPolicyTrainer(algo, env, log_mode=["csv"],
                          logdir="./logs", epochs=100)
trainer.train()
```
