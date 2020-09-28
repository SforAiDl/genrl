===================================
Deep Deterministic Policy Gradients
===================================

Objective
=========

Deep Deterministic Policy Gradients (DDPG) is a model-free actor-critic algorithm which deals with continuous action spaces. One simple approach of dealing with continuous 
action spaces can be discretizing the action space. However, this gives rise to several problems, the most significant being that the size of the action-space increases exponentially 
with the number of degrees of freedom. DDPG builds up on *Deterministic Policy Gradients* to learn deterministic policies in high-dimensional continuous action-spaces.

Algorithms Details
==================

Deterministic Policy Gradient
-----------------------------

In cases with continuous action-spaces, using Q-learning like approach (greedy policy improvement) to learn deterministic policies is not feasible since it involves selecting the action with the maximum action value function 
at every step and it is not possible to check the action value for every possible action in case of continuous action spaces. 

.. math::

    \mu^{k+1}(s) = argmax_a Q^{\mu^{k}}(s, a)

This problem can be solved by considering the fact that a policy can be improved by moving it in the direction of increasing action-value function:

.. math::

    \nabla_{\theta^{\mu}}J = \mathbb{E}_{s_t \sim \rho^{\beta}}[\nabla_{\theta^{\mu}}Q(s, a \vert \theta^{Q}) \vert_{s=s_t, a=\mu(s_t, \theta^{\mu})}]

Action Selection
----------------

To ensure sufficient exploration, noise is added to the action selected using the current policy. The noise is sampled from a noise process :math:`\mathcal{N}` :

.. math::

    \mu'(s_t) = \mu(s_t \vert \theta_t^{\mu}) + \mathcal{N}

:math:`\mathcal{N}` can be chosen to suit the environment (for eg. Ornstein-Uhlenbeck process, Gaussian noise, etc.)

.. literalinclude:: ../../../../../genrl/agents/deep/base/offpolicy.py
    :lines: 156-179
    :lineno-start: 156

Experience Replay
-----------------

Similar to DQNs, DDPG being an off-policy algorithm, makes use of *Replay Buffers*. Whenever a transition :math:`(s_t, a_t, r_t, s_{t+1})` is encountered, it is stored into the replay buffer. Batches of these transitions are
sampled while updating the network parameters. This helps in breaking the strong correlation between the updates that would have been present had the transitions been trained and discarded immediately after they are encountered
and also helps to avoid the rapid forgetting of the possibly rare transitions that would be useful later on.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

Update the Value and Policy Networks
------------------------------------

DDPG makes use of target networks for the actor(policy) and the critic(value) networks to stabilise the training. The Q-network is update using TD-learning updates. The target and the loss function for the same are defined as:

.. math::

    L(\theta^{Q}) = \mathbb{E}_{(s_t \sim \rho^{\beta}, a_t \sim \beta, t_t \sim R)}[(Q(s_t, a_t \vert \theta^{Q}) - y_t)^{2}]

.. math::

    y_t = r(s_t, a_t) + \gamma Q_targ(s_{t+1}, \mu_targ(s_{t+1}) \vert \theta^{Q})

Buliding up on Deterministic Policy Gradients, the gradient of the policy can be determined using the action-value function as

.. math::

    \nabla_{\theta^{\mu}} J = \mathbb{E}_{s_t \sim \rho^{\beta}}[\nabla_{\theta^{\mu}}Q(s, a \vert \theta^{Q})\vert_{s=s_t, a=\mu(s_t \vert \theta^{\mu})}]

.. math::

    \nabla_{\theta^{\mu}} J  = \mathbb{E}_{s_t \sim \rho^{\beta}}[\nabla_a Q(s, a \vert \theta^{Q}) \vert_{s=s_t, a=\mu(s_t)}\nabla_{\theta_\mu}\mu(s \vert \theta^{\mu}) \vert_{s=s_t}]

The target networks are updated at regular intervals

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 145-203
    :lineno-start: 145

Training through the API
========================

.. code-block:: python

    from genrl.agents import DDPG
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("MountainCarContinuous-v0")
    agent = DDPG("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()







