Gradients
=========

Using Gradient Method on a Bernoulli Multi-Armed Bandit
-------------------------------------------------------

For an introduction to Multi Armed Bandits, refer to :ref:`bandit_overview`

This method is different compared to others. In other methods, we
explicity attempt to estimate the 'value' of taking an action (its
quality) whereas in this method we approach the problem in a different
way. Here, instead of estimating how good an action is through its
quality, we only care about its preference of being selected compared to
other actions. We denote this preference by :math:`H_t(a)`. The larger
the preference of an action 'a', more are the chances of it being
selected, but this preference has no interpretation in terms of the
reward for that action. Only the relative preference is important.

The action probabilites are related to these action preferences
:math:`H_t(a)` by a softmax function. The probability of taking action
:math:`a_j` is given by:

.. math:: P(a_j) = \frac{e^{H_t(a_j)}}{\sum_{i=1}^A e^{H_t(a_i)}} = \pi_t(a_j)

where, A is the total number of actions and :math:`\pi_t(a)` is the
probability of taking action 'a' at timestep 't'.

We initialise the preferences for all the actions to be 0, meaning
:math:`\pi_t(a) = \frac{1}{A}` for all actions.

After computing :math:`\pi_t(a)` for all actions at each timestep, the
action is sampled using this probability. Then that action is performed
and based on the reward we get, we update our preferences.

The update rule bacially performs stochastic gradient ascent:

:math:`H_{t+1}(a_t) = H_t(a_t) + \alpha (R_t - \bar{R_t})(1-\pi_t(a_t))`,
for :math:`a_t`: action taken at time 't'

:math:`H_{t+1}(a) = H_t(a) - \alpha (R_t - \bar{R_t})(\pi_t(a))` for
rest of the actions

where, :math:`\alpha` is the step size, :math:`R_t` is the reward
obtained at time 't' and :math:`\bar{R_t}` is the mean reward obtained
upto time t. If current reward is larger than the mean reward, we
increase our preference for that action taken at time 't'. If it is
lower than the mean reward, we decrease our preference for that action.
The preferences for the rest of the actions are updated in the opposite
direction.

For a more detailed mathematical analysis and derivation of the update
rule, refer to chapter 2 of Sutton & Barto.

Code to use the Gradient method on a Bernoulli Multi-Armed Bandit:

.. code:: python

    import gym
    import numpy as np

    from genrl.bandit import BernoulliMAB, GradientMABAgent, MABTrainer

    bandits = 10
    arms = 5

    reward_probs = np.random.random(size=(bandits, arms))
    bandit = BernoulliMAB(bandits, arms, reward_probs, context_type="int")
    agent = GradientMABAgent(bandit, alpha=0.1, temp=0.01)

    trainer = MABTrainer(agent, bandit)
    trainer.train(timesteps=10000)

More details can be found in the docs for
`BernoulliMAB <../../../api/bandit/genrl.core.bandit.html#genrl.core.bandit.bernoulli_mab.BernoulliMAB>`__,
`BayesianUCBMABAgent <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.gradient>`__
and
`MABTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.
