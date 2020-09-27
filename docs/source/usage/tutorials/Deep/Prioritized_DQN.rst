===========================
Prioritized Deep Q-Networks
===========================

Objective
=========

The main motivation behind using prioritized experience replay over uniformly sampled experience replay stems from the fact that an agent may be able to learn more 
from some transitions than others. In uniformly sampled experience replay, some transitions which might not be very useful for the agent or that might be redundant will
be replayed with the same frequency as those having more learning potential. Prioritized experience replay solves this problem by replaying more useful transitions more frequently.

The loss function for prioritized DQN is defined as 

.. math::

    E_{(s, a, s', r, p) \sim D}[r + \gamma max_{a'} Q(s', a';\theta_{i}^{-}) - Q(s, a; \theta_i)]^2

Algorithm Details
=================

Epsilon-Greedy Action Selection
-------------------------------

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/dqn.py
    :lines: 97-129
    :lineno-start: 97

The action exploration is stochastic wherein the greedy action is chosen with a probability of :math:`1 - \epsilon` and rest of the time, we sample the action randomly. During evaluation, we use only greedy actions to judge how well the agent performs.

Prioritized Experience Replay
-----------------------------

The replay buffer is no longer uniformly sampled, but is sampled according to the *priority* of a transition. Transitions with greater scope of learning are assigned a higher priorities.
Priority of a particular transition is decided using the TD-error since the measure of the magnitude of the TD error can be interpreted as how unexpected the transition is.

.. math::

    \delta = R + \gamma max_{a'} Q(s', a';\theta_{i}^{-}) - Q(s, a; \theta_i)

The transition with the maximum TD-error is given the maximum priority. Every new transition is given the highest priority to ensure that each transition is considered at-least once.

Stochastic Prioritization
+++++++++++++++++++++++++

Sampling transition greedily has some disadvantages such as transitions having a low TD-error on the first replay might not be sampled ever again, higher chances of overfitting since only a small set of transitions with high priorities are replayed over and
over again and sensitivity to noise spikes. To tackle these problems, instead of sampling transitions greedily everytime, we use a stochastic approach wherein each transition is assigned a certain probability with which it is sampled. The sampling probability is defined as

.. math::

    P(i) = \frac{p_i^{\alpha}}{\Sigma_k p_k^{\alpha}}

where :math:`p_i > 0` is the priority of transition :math:`i`. :math:`\alpha` determines the amount of prioritization done. The priority of the transition can be defined in the following two ways:

* :math:`p_i = |\delta_i| + \epsilon`
* :math:`p_i = \frac{1}{rank(i)}`

where :math:`\epsilon` is a small positive constant to ensure that the sampling probability is not zero for any transition and :math:`rank(i)` is the rank of the transition when the replay buffer is sorted with respect to priorities.

We also use importance sampling (IS) weights to correct certain bais introduced by prioritized experience replay. 

.. math::
    w_i = (\frac{1}{N} \frac{1}{P(i)})^{\beta}

Update the Q-value Networks
---------------------------

The importance sampling weights can be folded into the Q-learning update by using :math:`w\delta_i` instead of :math:`\delta_i`. Once our Replay Buffer has enough experiences, we start updating the Q-value networks in the following code according to the above objective.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 145-203
    :lineno-start: 145

Training through the API
========================

.. code-block:: python

    from genrl.agents import PrioritizedReplayDQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("CartPole-v0")
    agent = PrioritizedReplayDQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()







