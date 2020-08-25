# Custom Policy Networks

GenRL provides default policies for images (CNNPolicy) and for other types of inputs(MlpPolicy).
Sometimes, these default policies may be insuffiecient for your problem, or you may want more control over the policy definition, and hence require a custom policy.  

The following code tutorial runs through the steps to use a custom policy depending on your problem. 

Import the required libraries (eg. torch, torch.nn) and from GenRL, the algorithm (eg VPG), the trainer (eg. OnPolicyTrainer), the policy to be modified (eg. MlpPolicy)
```python
# The necessary imports
import torch
import torch.nn as nn

from genrl import VPG
from genrl.core.policies import MlpPolicy
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer

```

Then define a `custom_policy` class that derives from the policy to be modified (in this case, the `MlpPolicy`)
```Python
# Define a custom MLP Policy
class custom_policy(MlpPolicy):
    def __init__(self, state_dim, action_dim, hidden, **kwargs):
        super().__init__(state_dim, action_dim, hidden)
        self.action_dim = action_dim
        self.state_dim = state_dim
```
The above class modifies the MlpPolicy to have the desired number of hidden layers in the MLP Neural network that parametrizes the policy.
This is done by passing the variable hidden explicitly (default`hidden = (32, 32)`). The `state_dim` and `action_dim` variables stand for the dimensions of the state_space and the action_space, and are required to construct the neural network with the proper input and output shapes for your policy, given the environment.


In some cases, you may also want to redefine the policy used completely and not just customize and existing policy. This can be done by creating a new custom policy class that inhierits the BasePolicy class.
The BasePolicy class is a basic implementation of a general policy, with a `forward` and a `get_action` method. The forward method maps the input state to the action probabilities,
and the `get_action` method selects an action from the given action probabilities (for both continuous and discrete action_spaces)   

Say you want to parametrize your policy by a Neural Network containing LSTM layers followed my MLP layers. This can be done as follows:

```python
# Define a custom LSTM policy from the BasePolicy class
class custom_policy(BasePolicy):
    def __init__(self, state_dim, action_dim, hidden,
                 discrete=True, layer_size=512, layers=1, **kwargs):
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
```

Finally, it's time to train the custom policy. Define the environment to be trained on (`CartPole-v0` in this case), and the `state_dim` and `action_dim` variables.

```Python
# Initialize an environment
env = VectorEnv("CartPole-v0", 1)

# Initialize the custom Policy
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = custom_policy(state_dim=state_dim, action_dim=action_dim,
                        hidden = (64, 64))
```

Then the algorithm is initialised with the custom policy defined, and the OnPolicyTrainer trains in with logging for better inference.
```Python
algo = VPG(policy, env)

# Initialize the trainer and start training 
trainer = OnPolicyTrainer(algo, env, log_mode=["csv"],
                          logdir="./logs", epochs=100)
trainer.train()
```
