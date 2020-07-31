### Using A2C on "CartPole-v0"

```python
import gym

from genrl import A2C
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("CartPole-v0")
agent = A2C('mlp', env, gamma=0.9, lr_policy=0.01, lr_value=0.1, layers=(32,32), rollout_size=2048)
trainer = OnPolicyTrainer(agent, env, log_mode=['stdout', 'tensorboard'], log_key="Episode")
trainer.train()
```

### Using A2C on atari env - "Pong-v0"

```python
import gym

from genrl import A2C
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("Pong-v0", env_type = "atari")
agent = A2C('cnn', env, gamma=0.99, lr_policy=0.01, lr_value=0.1, layers=(32,32), rollout_size=2048)
trainer = OnPolicyTrainer(agent, env, log_mode=['stdout', 'tensorboard'], log_key="timestep")
trainer.train()
```

Hyperparameters and arguments for the A2C agent:

https://github.com/SforAiDl/genrl/blob/master/genrl/deep/agents/a2c/a2c.py 

Hyperparameters and arguments for the OnPolicyTrainer:

https://github.com/SforAiDl/genrl/blob/master/genrl/deep/common/trainer.py
