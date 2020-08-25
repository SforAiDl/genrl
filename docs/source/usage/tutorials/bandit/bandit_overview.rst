.. _bandit_overview:

Multi Armed Bandit Overview
===========================

Training an EpsilonGreedy agent on a Bernoulli Multi Armed Bandit
-----------------------------------------------------------------

Multi armed bandits is one of the most basic problems in RL. Think of it
like this, you have 'n' levers in front of you and each of these levers
will give you a different reward. For the purposes of formalising the
problem the reward is written down in terms of a reward function i.e.,
the probability of getting a reward when a lever is pulled.

Suppose you try out one of the levers and get a positive reward. What do
you do next? Should you just keep pulling that lever every time or think
what if there might be a better reward to pulling one of the other
levers? This is the exploration - exploitation dilemma.

*Exploitation* - Utilise the information you have gathered till now, to
make the best decision. In this case, after 1 try you know a lever is
giving you a positive reward and you just *exploit* it further. Since
you do not care about other arms if you keep *exploiting*, it is known
as the greedy action.

*Exploration* - You explore the untried levers in an attempt to maybe
discover another one which has a higher payout than the one you
currently have some knowledge about. This is exploring all your options
without worrying about the short-term rewards, in hope of finding a
lever with a bigger reward, in the long run.

You have to use an algorithm which correctly trades off exploration and
exploitation as we do not want a 'greedy' algorithm which only exploits
and does not explore at all, because there are very high chances that it
will converge to a sub-optimal policy. We do not want an algorithm that
keeps exploring either as this would lead to sub-optimal rewards inspite
of knowing the best action to be taken. In this case, the optimal policy
will be to always pull the lever with the highest reward, but at the
beginning we do not know the probability distribution of the rewards.

So, we want a policy which explores actively at the beginning, building
up an estimate for the reward values(defined as *quality*) of all the
actions, and then exploiting that from that time onwards.

A Bernoulli Multi-Armed Bandit has multiple arms with each having a
different bernoulli distribution over its reward. Basically each arm has
a probabilty associated with it which is the probability of getting a
reward if that arm is pulled. Our aim is to find the arm which has the
highest probabilty, thus giving us the maximum return.

Notation:

:math:`Q_t(a)`: Estimated quality of action 'a' at timestep 't'.

:math:`q(a)`: True value of action 'a'.

We want our estimate :math:`Q_t(a)` to be as close to the true value
:math:`q(a)` as possible, so we can make the correct decision.

Let the action with the maximum quality be :math:`a^*`:s

.. math:: q^* = q(a^*)

Our goal is to find this :math:`q^*`.

The 'regret function' is defined as the sum of 'regret' accumulated over
all timesteps. This regret is the cost of not choosing the optimal arm
and instead of exploring. Mathematically it can be written as:

.. math:: L = E[\sum_{t=0}^T q^* - Q_t(a)]

Some policies which are effective at exploring are: 1. `Epsilon
Greedy <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.epsgreedy>`__
2. `Gradient
Algorithm <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.gradient>`__
3. `UCB(Upper Confidence
Bound) <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.ucb>`__
4.
`Bayesian <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.bayesian>`__
5. `Thompson
Sampling <../../../bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.thompson>`__

Epsilon Greedy is the most basic exploratory policy which follows a
simple principle to balance exploration and exploitation. It 'exploits'
the current knowledge of the bandit most of the times, i.e. takes the
action with the largest q value. But with a small probability epsilon,
it also explores a random action. The value of epsilon signifies how
much you want the agent explore. Higher the value, the more it explores.
But remember you do not want an agent to explore too much even after it
has a pretty confident estimate of the reward function, so the value of
epislon should neither be too high nor too low!

For the bandit, you can set the number of bandits, number of arms, and
also reward probabilities of each of these arms seperately.

Code to train an Epsilon Greedy agent on a Bernoulli Multi-Armed Bandit:

.. code:: python

    import gym
    import numpy as np

    from genrl.bandit import BernoulliMAB, EpsGreedyMABAgent, MABTrainer

    reward_probs = np.random.random(size=(bandits, arms))
    bandit = BernoulliMAB(arms=5, reward_probs=reward_probs, context_type="int")
    agent = EpsGreedyMABAgent(bandit, eps=0.05)

    trainer = MABTrainer(agent, bandit)
    trainer.train(timesteps=10000)

More details can be found in the docs for
`BernoulliMAB <../../../api/bandit/genrl.core.bandit.html#genrl.core.bandit.bernoulli_mab.BernoulliMAB>`__,
`EpsGreedyMABAgent <../../../api/bandit/genrl.agents.bandits.multiarmed.html#module-genrl.agents.bandits.multiarmed.epsgreedy>`__,
`MABTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.

You can also refer to the book "Reinforcement Learning: An
Introduction", Chapter 2 for further information on bandits.
