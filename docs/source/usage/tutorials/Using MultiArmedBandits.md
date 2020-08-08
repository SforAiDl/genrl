### Using UCB agent on a Bernoulli Multi Armed Bandit

```python
import gym

from genrl.bandit import BernoulliMAB, MABTrainer, UCBMABAgent

bandit = BernoulliMAB(bandits=10, arms=5, context_type="int")
agent = UCBMABAgent(bandit, confidence=1.0)

trainer = MABTrainer(agent, bandit)
trainer.train(timesteps=10000)
```

Implementation of the bandit:
https://github.com/SforAiDl/genrl/blob/master/genrl/bandit/bandits/multi_armed_bandits/bernoulli_mab.py

Implementation of the agent:
https://github.com/SforAiDl/genrl/blob/master/genrl/bandit/agents/mab_agents/ucb.py
