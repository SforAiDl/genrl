UCB
===

Training a UCB algorithm on a Bernoulli Multi-Armed Bandit
----------------------------------------------------------

For an introduction to Multi Armed Bandits, refer to :ref:`bandit_overview`

The UCB algorithm follows a basic principle - 'Optimism in the face of
uncertainty'. What this means is that we should always select the action
whose reward we are most uncertain of. We quantify the uncertainty of
taking an action by calculating an upper bound of the quality(reward)
for that action. We then select the greedy action with respect to this
upper bound.

Hoeffding's inequality:

.. math::  P[q(a) > Q_t(a) + U_t(a)] \le e ^ {-2 N_t(a) U_t(a)^2}

,

q(a) is the quality of that action,

:math:`Q_t(a)` is the estimate of the quality of action 'a' at time 't',

:math:`U_t(a)` is the upper bound for uncertainty for that action at
time 't',

:math:`N_t(a` is the number of times action 'a' has been selected

.. math::  e ^ {-2 N_t(a) U_t(a)^2} = t^{-4} 

.. math::  U_t(a) = \sqrt{\frac{2 log t}{N_t(a)}} 

Action taken: a = argmax\ :math:`(Q_t(a) + U_t(a))`

As we can see, the less an action has been tried, more the uncertainty
is (due to :math:`N_t(a)` being in the denominator), which leads to that
action having a higher chance of being explored. Also, theoretically, as
:math:`{N_t(a)}` goes to infinity, the uncertainty decreases to 0 giving
us the true value of the quality of that action: q(a). This allows us to
'exploit' the greedy action :math:`a^*` from then.

Code to train a UCB agent on a Bernoulli Multi-Armed Bandit:

.. code:: python

    import gym
    import numpy as np

    from genrl.bandit import BernoulliMAB, MABTrainer, UCBMABAgent

    bandits = 10
    arms = 5

    reward_probs = np.random.random(size=(bandits, arms))
    bandit = BernoulliMAB(bandits, arms, reward_probs, context_type="int")
    agent = UCBMABAgent(bandit, confidence=1.0)

    trainer = MABTrainer(agent, bandit)
    trainer.train(timesteps=10000)

More details can be found in the docs for
`BernoulliMAB <../../../api/bandit/genrl.core.bandit.html#genrl.core.bandit.bernoulli_mab.BernoulliMAB>`__,
`UCB <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.ucb>`__
and
`MABTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.
