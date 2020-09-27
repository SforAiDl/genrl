Categorical Deep Q-Networks
===========================

Objective
=========

The main objective of Categorical Deep Q-Networks is to learn the distribution of Q-values as unlike to other variants of Deep Q-Networks where the goal is
is to approximate the *expectations* of the Q-values as closely as possible. In complicated environments, the Q-values can be stochastic and in that case, 
simply learning the expectation of Q-values will not be able to capture all the information needed (for eg. variance of the distribution) to make an optimal 
decision. 

Distributional Bellman
======================

The bellman equation can be adapted to this form as 

.. math::

    Z(x, a) \stackrel{D}{\eq} R(x, a) + \gamma Z(x', a')

where :math:`Z(s, a)` (the value distribution) and :math:`R(s, a)` (the reward distribution) are now probability distributions. The equality or similarity of two distributions can be effectivelyevaluated using 
the Kullback-Leibler(KL) - divergence or the cross-entropy loss. 

.. math::

    Q^{\pi}(x, a) \coloneqq \mathbb{E} Z^{\pi}(x, a) = \mathbb{E}\left[\sum_{t=0}^{\inf} \gamma^{t} R(x_t, a_t)\right]
    z \sim P(\odot \vert x_{t-1}, a_{t-1}). a_t \sim \pi(\odot \vert x_t), x_0 = x, a_0 =a

The transition operator :math:`P^\pi : \Zstroke \rightarrow \Zstroke` and the bellman operator :math:`\mathcal{T} : \Zstroke \rightarrow \Zstroke`
can be defined as 

.. math::

    P^{\pi}Z(x, a) \stackrel{D}{\coloneqq} Z(X', A') ; X' \sim P(\odot \vert x, a), A' \sim \pi(\odot \vert X')

.. math::

    \mathcal{T}^{\pi}Z(x, a) \stackrel{D}{\coloneqq} R(x, a)+ \gamma P^{\pi}Z(x, a)

Algorithm Details
=================

Parametric Distribution
-----------------------

Categorical DQN uses a discrete distribution parameterized by a set of supports/*atoms* (discrete values) to model the value distribution.
This set of atoms is determined as 

.. math::

    {\mathcal{z}_i = V_{MIN} + i \nabla \mathcal{z} : 0 \leq i < N}; \nabla \mathcal{z} \coloneqq \frac{V_{MAX} - V_{MIN}}{N - 1}

where :math:`N \in \mathbb{N}` and :math:`V_{MAX}, V_{MIN} \in \mathbb{R}` are the distribution parameters. The probability of each atom is modeled as

.. math::

    Z_\theta(x, a) = \mathcal{z}_i w.p. p_i(x, a) \coloneqq \frac{\exp{\theta_i(x, a)}}{\sum_j \exp{\theta_j(x, a)}}

Action Selection
----------------

GenRL uses greedy action selection for categorical DQN wherein the action with the highest Q-values for all discrete regions is selected.

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/utils.py
    :lines: 65-86
    :lineno-start: 65

Experience Replay
-----------------

Categorical DQN like other DQNs uses *Replay Buffer* like other off-policy algorithms. Whenever a transition :math:`(s_t, a_t, r_t, s_{t+1})` is encountered, it is stored into the replay buffer. Batches of these transitions are
sampled while updating the network parameters. This helps in breaking the strong correlation between the updates that would have been present had the transitions been trained and discarded immediately after they are encountered
and also helps to avoid the rapid forgetting of the possibly rare transitions that would be useful later on.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

Projected Bellman Update
------------------------

The sample bellman update :math:`\hat{\mathcal{T}}Z_\theta` is projected onto the support of :math:`Z_\theta` for the update as shown in the 
figure below. The bellman update for each atom :math:`j` can be calculated as 

.. math::

    \hat{\mathcal{T}}\mathcal{z_j} \coloneqq r + \gamma \mathcal{z_j}

and then it's probability :math:`\mathcal{p_j}(x', \pi{x'})` is distributed to the neighbours of the update. Here, :math:`(x, a, r, x')` is a sample transition.
The :math:`i^{th}` component of the projected update is calculated as 

.. math::

    (\Phi \hat{\mathcal{T}} Z_\theta(x, a))_i = \mathlarger{\sum}_{j=0}^{N-1}\left [1 - \frac{\mid \left [\hat{\mathcal{T}}\mathcal{z_j}\right]_{V_{MIN}}^{V_{MAX}} - \mathcal{z_i} \mid}{\Delta \mathcal{z}}\right]_{0}^{1} \mathcal{p_j}(x', \pi(x'))

The loss is calculated using KL divergence (cross entropy loss). This is also known as the **Bernoulli algorithm**

.. math::

    D_{KL}(\Phi\hat{\mathcal{T}}Z_\tilde{\theta}(x, a) || Z_\theta (x, a))

|

.. image:: static/Categorical_DQN.png

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/utils.py
    :lines: 120-185
    :lineno-start: 120


Training through the API
========================

.. code-block:: python

    from genrl.agents import CategoricalDQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("CartPole-v0")
    agent = CategoricalDQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()
