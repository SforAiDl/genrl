# Using Thompson Sampling

## Using Thompson Sampling on a Bernoulli Multi-Armed Bandit

For an introduction to Multi Armed Bandits, refer to [Tutorial On Bandits](https://genrl.readthedocs.io/en/latest/usage/tutorials/Tutorial_on_bandits.html).

Thompson Sampling is one of the best methods for solving the Bernoulli multi-armed bandits problem. It is a 'sample-based probability matching' method.

We initially _assume_ an initial distribution(prior) over the quality of each of the arms. We can model this prior using a Beta distribution, parametrised by alpha($\alpha$) and beta($\beta$).

$$PDF = \frac{x^{\alpha - 1} (1-x)^{\beta -1}}{B(\alpha, \beta)}$$

Let's just think of the denominator as some normalising constant, and focus on the numerator for now. We initialise $\alpha$ = $\beta$ = 1. This will result in a uniform distribution over the values (0, 1), making all the values of quality from 0 to 1 equally probable, so this is a fair initial assumption. Now think of $\alpha$ as the number of times we get the reward '1' and $\beta$ as the number of times we get '0', for a particular arm. As our agent interacts with the environment and gets a reward for pulling any arm, we will update our prior for that arm using Bayes Theorem. What this does is that it gives a posterior distribution over the quality, according to the rewards we have seen so far.

At each timestep, we sample the quality: $Q_t(a)$ for each arm from the posterior and select the sample with the highest value. The more an action is tried out, the narrower is the distribution over its quality, meaning we have a confident estimate of its quality (q(a)). If an action has not been tried out that often, it will have a more wider distribution (high variance), meaning we are uncertain about our estimate of its quality (q(a)). This wider variance of an arm with an uncertain estimate creates opportunities for it to be picked during sampling.

As we experience more successes for a particular arm, the value of $\alpha$ for that arm increases and similiarly, the more failures we experience, the value of $\beta$ increases. Higher the value of one of the parameters as compared to the other, the more skewed is the distribution in one of the directions. For eg. if $\alpha$ = 100 and $\beta$ = 50, we have seen considerably more successes than failures for this arm and so our estimate for its quality should be >0.5. This will be reflected in the posterior of this arm, i.e. the mean of the distribution, characterised by $\frac{\alpha}{\alpha + \beta}$ will be 0.66, which is >0.5 as we expected.

Code to use Thompson Sampling on a Bernoulli Multi-Armed Bandit:

```python
import gym
import numpy as np

from genrl.bandit import BernoulliMAB, MABTrainer, ThompsonSamplingMABAgent

bandits = 10
arms = 5
alpha = 1.0
beta = 1.0

reward_probs = np.random.random(size=(bandits, arms))
bandit = BernoulliMAB(bandits, arms, reward_probs, context_type="int")
agent = ThompsonSamplingMABAgent(bandit, alpha, beta)

trainer = MABTrainer(agent, bandit)
trainer.train(timesteps=10000)
```

More details can be found in the docs for [BernoulliMAB](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.bandits.multi_armed_bandits.html#genrl.bandit.bandits.multi_armed_bandits.bernoulli_mab.BernoulliMAB), [ThompsonSamplingMABAgent](https://genrl.readthedocs.io/en/latest/api/bandit/genrl.bandit.agents.mab_agents.html#module-genrl.bandit.agents.mab_agents.thompson) and [MABTrainer](https://genrl.readthedocs.io/en/latest/api/common/bandit.html#module-genrl.bandit.trainer).
