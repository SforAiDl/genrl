=======================
NoisyNet Deep Q-Network
=======================

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

The action selection is no longer greedy since the exploration is driven by the noise in the neural network layers. The action selection is done greedily.

Noisy Parameters
----------------

A noisy parameter :math:`\theta` is defined as:

.. math::

    \theta \coloneqq \mu + \Sigma \odot \epsilon

where :math:`\Sigma` and :math:`\mu` are vectors of trainable parameters and :math:`\epsilon` is a vector of zero mean noise.

