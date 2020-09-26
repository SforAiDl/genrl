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

Action-Value Networks
---------------------

SAC learns a ploicy :math:`\pi_\theta` and two Q functions :math:`Q_{\phi_1}, Q_{\phi_2}` and their target networks concurrently. The two Q-functions are learned in a fashion similar to TD3 where a common target is considered for both the Q functions and 
*Clipped Double Q-learning* is used to train the network. However, unlike TD3, the next-state actions used in the target are calculated using the current policy. Since, the optimisation objective also involves maximising the entropy, 
the new Q-value can be expressed as

.. math::

    Q^{\pi}(s, a) = \mathbb{E}_(s' \sim P, a' \sim \pi)[R(s, a, s') + \gamma(Q^{\pi}(s', a') + \alpha\mathcal{H}(\pi(\cdot \vert s')))]
                  = \mathbb{E}_(s' \sim P, a' \sim \pi)[R(s, a, s') + \gamma(Q^{\pi}(s', a') + \alpha log\pi(a' \vert s'))]

Thus, the action-value for one state-action pair can be approximated as

.. math::

    Q^{\pi}(s, a) \approx r + \gamma(Q^{\pi}(s', \tilde{a'}) - \alpha log \pi(\tilde{a' } \vert s'))

where :math:`\tilde{a'}` (action taken in next state) is sampled from the policy.

Experience Replay
-----------------

SAC also uses *Replay Buffer* like other off-policy algorithms. Whenever a transition :math:`(s_t, a_t, r_t, s_{t+1})` is encountered, it is stored into the replay buffer. Batches of these transitions are
sampled while updating the network parameters. This helps in breaking the strong correlation between the updates that would have been present had the transitions been trained and discarded immediately after they are encountered
and also helps to avoid the rapid forgetting of the possibly rare transitions that would be useful later on.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91


Q-Network Optimisation
----------------------

Just like TD3, SAC uses *Clipped Double Q-Learning* to calculate the target values for the Q-value network

.. math::

    y^{t}(r, s', d) = r + \gamma (min_{j=1,2}Q_{\phi_{targ,j}}(s', \tilde{a'}) - \alpha log \pi_{\theta}(\tilde{a'} \vert s'))

where :math:`\tilde{a'}` is sampled from the policy. The loss function can then be defined as

.. math::

    L(\phi_{i}, \mathcal{D}) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}}[(Q_{\phi_{i}}(s, a) - y^t(r, s', d))^2]


Action Selection and Policy Optimisation
----------------------------------------

The main aim of policy optimisation will be maximise the value function which in this case can be defined as 

.. math::

    V^{\pi}(s) = \mathbb{E}_{a \sim \pi}[Q^{\pi}(s, a) - log \pi(a \vert s)]

In SAC, a **reparameterisation trick** is used to sample actions from the policy to ensure that sampling from the policy is  a differentiable process.
The policy is now parameterised as 

.. math::

    \tilde{a'}_t = \mathcal{f}_\theta(\xi_t; s_t)

.. math::
    
    \tilde{a'}_{\theta}(s, \xi) = tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \xi)

.. math::

    \xi \sim \mathcal{N}(0, 1)

The maximisation objective is now defined as 

.. math::

    max_{\theta} \mathbb{E}_{s \sim \mathcal{D}, \xi \sim \mathcal{N}}[min_{j=1,2}Q_{\phi_j}(s, \tilde{a}_\theta(s, \xi)) - \alpha log \pi_{\theta}(\tilde{a}_{\theta}(s, \xi) \vert s)]

Training through the API
========================

.. code-block:: python

    from genrl.agents import SAC
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("MountainCarContinuous-v0")
    agent = SAC("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=4000)
    trainer.train()
    trainer.evaluate()







