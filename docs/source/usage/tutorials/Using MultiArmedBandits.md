### Using UCB agent on a Bernoulli Multi Armed Bandit

Multi armed bandits is one of the most basic problems in RL. Think of it like this, you have 'n' levers in front of you and each of these levers has a different reward function (the probability of getting a reward when a lever is pulled). Suppose you try out one of the levers and get a positive reward. What do you do next? Do you just keep pulling that lever every time or think what if there might be a better reward to pulling one of the other levers? This is the exploration - exploitation dilemma.

Exploitation - You use the information you have gathered till now, to make the best decision. In this case, after 1 try you know a lever is giving you a positive reward and you just 'exploit' it further(known as the greedy action).

Exploration - You explore the untried levers in an attempt to maybe discover another one which has a higher payout than the one you currently have some knowledge about. This is exploring all your options without worrying about the maximum reward instantly, in hope of finding a lever with a bigger reward, in the long run.

You have to use an algorithm which correctly trades off exploration and exploitation as we do not want a 'greedy' algorithm which only exploits and does not explore at all, because there are very high chances that it will converge to a sub-optimum policy. In this case, the optimum policy will be to always pull the lever with the highest reward. So, we want a policy which explores actively at the beginning, finding the best arm, and then exploiting that from that time onwards.

Some policies which are effective at exploring are: 
1. [Epsilon Greedy](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.epsgreedy)
2. [Gradient Algorithm](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.gradient)
3. [UCB(Upper Confidence Bound)](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.ucb)
4. [Bayesian](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.bayesian)
5. [Thompson Sampling](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.thompson)

We will be demonstrating how to use the UCB algorithm on a Bernoulli Bandit Multi-Armed Bandit.

A Bernoulli Multi-Armed Banit has multiple arms with each having a different bernoulli distribution over its reward. Basically each arm has a probabilty associated with it which is the probability of getting a reward if that arm is pulled. Our aim is to find the arm which has the highest probabilty, thus givng us the maximum return.

The UCB algorithm follows a basic principle - 'Optimism in the face of uncertainty'. What this means is that we should always select the action whose reward we are most uncertain of. We quantify the uncertainty of taking an action by calculating an upper bound of the quality(reward) for that action. We then select the greedy action with respect to this upper bound. As the number of times an arm is pulled increases, the uncertainty in its quality decreases and thus we should theoretically converge to an optimum policy with zero uncertainty for all arms, allowing us to 'exploit' the greedy action based on the true quality of the arms from then.

```python
import gym

from genrl.bandit import BernoulliMAB, MABTrainer, UCBMABAgent

bandit = BernoulliMAB(bandits=10, arms=5, context_type="int")
agent = UCBMABAgent(bandit, confidence=1.0)

trainer = MABTrainer(agent, bandit)
trainer.train(timesteps=10000)
```

More details can be found in the docs for [BernoulliMAB](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.bandits.multi_armed_bandits.html#genrl.bandit.bandits.multi_armed_bandits.bernoulli_mab.BernoulliMAB), [UCB](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.ucb) and [MABTrainer](https://genrl.readthedocs.io/en/latest/api/common/bandit.html#module-genrl.bandit.trainer).
