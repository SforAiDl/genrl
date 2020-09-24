=================
Soft Actor-Critic
=================

Objective
=========

Deep Reinforcement Learning Algorithms suffer from two main problems : one being high sample complexity (large amounts of data needed) and the other being thier brittleness with respect to learning 
rates, exporation constants and other hyperparameters. Algorithms such as DDPG and Twin Delayed DDPG are used to tackle the challenge of high sample complexity in actor-critic frameworks with continuous 
action-spaces. However, they still suffer from brittle stability with respect to their hyperparameters. Soft-Actor Critic introduces a actor-critic framework for arrangements with continuous action spaces
wherein the standard objective of reinforcement learning, i.e., maximizing expected cumulative reward is augmented with an additional objective of entropy maximization which provides a substantial improvement 
in exploration and robustness. The objective can be mathematically represented as 

.. math::

    J(\pi) = \Sigma_{t=0}^{T}\gamma^t\mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}}[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \vert s_t))]

where :math:`\alpha` also known as the temperature parameter determines the relative importance of the entropy term against the reward, and thus
controls the stochasticity of the optimal policy and :math:`\mathcal{H}` represents the entropy function. The entropy of a random variable :math:`\mathcal{x}`
following a probability distribution :math:`P` is defined as 

.. math::

    \mathcal{H}(P) = \mathbb{E}_{\mathcal{x} \sim P}[-logP(\mathcal{x})]

Algorithm Details
=================

Soft Actor-Critic is mostly used in two variants depending on whether the temperature constant :math:`\alpha` is kept constant throughout the learning process or if it is learned as a parameter over the course of learning.
GenRL uses the latter one.

Action Selection
----------------

Action selection in SAC is done as 

.. math::

    \tilde{a}_\theta(s, \xi) = tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \xi), \xi \sim \mathcal{N}(0, 1)

Learning the Q-Values
------------------------------------

SAC learns a ploicy :math:`\pi_\theta` and two Q functions :math:`Q_{\phi_1}, Q_{\phi_2}` concurrently. The two Q-functions are learned in a fashion similar to TD3 where a common target is considered for both the Q functions and 
*Clipped Double Q-learning* is used to train the network. However, unlike TD3, the next-state actions used in the target are calculated using the current policy. Since, the optimisation objective also involves maximising the entropy, 
the new Q-value can be expressed as

.. math::

    Q^{\pi}(s, a) = \mathbb{E}_(s' \sim P, a' \sim \pi)[R(s, a, s') + \gamma(Q^{\pi}(s', a') + \alpha\mathcal{H}(\pi(\cdot \vert s')))]
                  = \mathbb{E}_(s' \sim P, a' \sim \pi)[R(s, a, s') + \gamma(Q^{\pi}(s', a') + \alpha log\pi(a' \vert s'))]

Experience Replay
-----------------

Updating the Value and the Policy Networks 
------------------------------------------

Training through the API
========================







