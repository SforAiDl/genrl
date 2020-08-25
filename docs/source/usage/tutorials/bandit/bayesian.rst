Bayesian
========

Using Bayesian Method on a Bernoulli Multi-Armed Bandit
-------------------------------------------------------

For an introduction to Multi Armed Bandits, refer to :ref:`bandit_overview`

This method is also based on the prinicple - 'Optimism in the face of
uncertainty', like
`UCB <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.ucb>`__.
We initially *assume* an initial distribution(prior) over the quality of
each of the arms. We can model this prior using a Beta distribution,
parametrised by alpha(\ :math:`\alpha`) and beta(\ :math:`\beta`).

.. math:: PDF = \frac{x^{\alpha - 1} (1-x)^{\beta -1}}{B(\alpha, \beta)}

Let's just think of the denominator as some normalising constant, and
focus on the numerator for now. We initialise :math:`\alpha` =
:math:`\beta` = 1. This will result in a uniform distribution over the
values (0, 1), making all the values of quality from 0 to 1 equally
probable, so this is a fair initial assumption. Now think of
:math:`\alpha` as the number of times we get the reward '1' and
:math:`\beta` as the number of times we get '0', for a particular arm.
As our agent interacts with the environment and gets a reward for
pulling any arm, we will update our prior for that arm using Bayes
Theorem. What this does is that it gives a posterior distribution over
the quality, according to the rewards we have seen so far.

This is quite similar to `Thompson
Sampling <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.thompson>`__.
But what is different here is that we explicity try to calculate the
uncertainty of a particular action by calculating the standard
deviation(\ :math:`\sigma`) of its posterior. We add this std. dev to
the mean of the posterior, giving us an *upper bound* of the quality of
that arm. At each timestep we select a greedy action based on this upper
bound we calculated.

.. math:: a_t = argmax(q_t(a) + \sigma_{q_t})

As we try out an action more and more, the standard deviation of the
posterior decreases, corresponding to a decrease in the uncertainty of
that action, which is exactly what we want. If an action has not been
tried that often, it will have a wider posterior, meaning higher chances
of it getting selected based on its upper bound.

Code to use Bayesian method on a Bernoulli Multi-Armed Bandit:

.. code:: python

    import gym
    import numpy as np

    from genrl.bandit import BayesianUCBMABAgent, BernoulliMAB, MABTrainer

    bandits = 10
    arms = 5
    alpha = 1.0
    beta = 1.0

    reward_probs = np.random.random(size=(bandits, arms))
    bandit = BernoulliMAB(bandits, arms, reward_probs, context_type="int")
    agent = BayesianUCBMABAgent(bandit, alpha, beta)

    trainer = MABTrainer(agent, bandit)
    trainer.train(timesteps=10000)

More details can be found in the docs for
`BernoulliMAB <../../../api/bandit/genrl.core.bandit.html#genrl.core.bandit.bernoulli_mab.BernoulliMAB>`__,
`BayesianUCBMABAgent <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.bayesian>`__
and
`MABTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.
