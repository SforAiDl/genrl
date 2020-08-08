# Vanilla Policy Gradient (VPG)

A barebones implementation of training an agent using the `VPG` class

```python
import gym #OpenAI Gym

from genrl import VPG
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("CartPole-v1")
agent = VPG('mlp', env)
trainer = OnPolicyTrainer(agent, env, epochs=200)
trainer.train()
```

Source Codes: \
`VPG` class  [here](https://github.com/SforAiDl/genrl/blob/9ce9716b18aab490e880c99cd530965db6bc4ba9/genrl/deep/agents/vpg/vpg.py)\
`OnPolicyTrainer` class [here](https://github.com/SforAiDl/genrl/blob/9ce9716b18aab490e880c99cd530965db6bc4ba9/genrl/deep/common/trainer.py#L406)\
`VectorEnv` method[here](https://github.com/SforAiDl/genrl/blob/9ce9716b18aab490e880c99cd530965db6bc4ba9/genrl/environments/suite.py#L16)

For more information on VPG you can go to the OpenAI documentation [here](https://spinningup.openai.com/en/latest/algorithms/vpg.html)