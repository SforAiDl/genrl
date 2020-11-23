===============================
Deep Q Networks with Noisy Nets
===============================

Objective
=========

NoisyNet DQN is a variant of DQN which uses fully connected layers with noisy parameters to drive exploration. Thus, the parametrised action-value function can now be seen as a random variable. The new loss function which 
needs to minimised is defined as:

.. math::

    E[E_{(x, a, r, y) \sim D}[r + \gamma max_{b \in A} Q(y, b, \epsilon'; \zeta^{-}) - Q(x, a, \epsilon; \zeta)]^{2}]

where :math:`\zeta` is a set of learnable parameters for the noise.

Algorithm Details
=================

Action Selection
----------------

The action selection is no longer epsilon-greedy since the exploration is driven by the noise in the neural network layers. The action selection is done greedily.

Noisy Parameters
----------------

A noisy parameter :math:`\theta` is defined as:

.. math::

    \theta := \mu + \Sigma \odot \epsilon

where :math:`\Sigma` and :math:`\mu` are vectors of trainable parameters and :math:`\epsilon` is a vector of zero mean noise. Hence, the loss function is now defined with respect to :math:`\Sigma` and :math:`\mu`
and the optimization now takes place with respect to :math:`\Sigma` and :math:`\mu`. :math:`\epsilon` is sampled from factorised gaussian noise.

Experience Replay
-----------------

Every transition occuring during the training is stored in a separate *Replay Buffer*

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

The transitions are later sampled in batches from the replay buffer for updating the network

Update the Q-Network
--------------------

Once enough number of transitions ae stored in the replay buffer, we start updating the Q-values according to the given objective. The loss function is defined in a fashion similar to a DQN. This allows
any new improvisations in training techniques of DQN such as Double DQN or NoisyNet DQN to be readily adapted in the dueling architechture.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 145-203
    :lineno-start: 145

Training through the API
========================

.. code-block:: python

    from genrl.agents import NoisyDQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("CartPole-v0")
    agent = NoisyDQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()
