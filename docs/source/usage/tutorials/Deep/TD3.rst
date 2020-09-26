=================
Twin Delayed DDPG
=================

Objective
=========

Similar to Deep Q-Networks, the problem of overestimation of the state values, occuring due to noisy function approximators and using the same function approximator for action selection and value estimation also persists in actor-critic 
algorithms with continuous action-spaces. Double DQN, the solution for this problem in Deep Q-Networks is not effective in actor-critic algorithms due to the slow rate of change of the policy. Twin Delayed DDPG (TD3) uses *Clipped Double Q-Learning* to 
address this problem. TD3 uses two Q function approximators and the loss function for each is given by 

.. math::

    L(\phi_{1}, \mathcal{D}) = E_{(s,a,r,s',d) \sim \mathcal{D}}[(Q_{\phi_{1}}(s, a) - y(r,s',d))^2]

.. math::

    L(\phi_{2}, \mathcal{D}) = E_{(s,a,r,s',d) \sim \mathcal{D}}[(Q_{\phi_{2}}(s, a) - y(r,s',d))^2]

Algorithm Details
=================

Clipped Double Q-Learning 
-------------------------

Double DQNs are not effective in actor-critic algorithms due to the slow change in the policy and the original double Q-Learning (although being somewhat effective) does not completely solve the problem of overestimation. To tackle this TD3 uses *Clipped Double Q-Learning*
Clipped Double Q-Learning proposes to upper bound the less biased critic network by the more biased one and hence no additional overestimation can be introdiced. Although, this may introduce underestimation, it is not much of a concern since underestimation errors don't propagate 
through policy updates. The target function calculated usign Clipped Double Q-Learning for the updates can be written as

.. math::

    y = r + \gamma min_{i=1,2}Q_{\theta_i'}(s', \pi_{\phi_1}(s'))

Both of the critic networks are updated using the loss functions mentioned above.

Experience Replay
-----------------

TD3 being an off-policy algorithm, makes use of *Replay Buffer*. Whenever a transition :math:`(s_t, a_t, r_t, s_{t+1})` is encountered, it is stored into the replay buffer. Batches of these transitions are
sampled while updating the network parameters. This helps in breaking the strong correlation between the updates that would have been present had the transitions been trained and discarded immediately after they are encountered
and also helps to avoid the rapid forgetting of the possibly rare transitions that would be useful later on.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

Target Policy Smoothing Regularization
--------------------------------------

TD3 adds noise to the target action to reduce the variance induced by function approximation error. This acts as a form of regularization which smoothens the changes in the action-values along changes in action

.. math::

    a = \pi_{\phi'}(s') + \epsilon

.. math::

    \epsilon \sim clip(\mathcal{N}(0, \sigma), -c, c)

Delayed Policy updates
----------------------

TD3 uses target networks similar to DDPG and DQNs for the two critics and the actors to stabilise learning. Apart from this, it also promotes updating the policy networks at a lower frequency as compared to the Q-networks to avoid divergent behaviour for the policy. The paper 
recommends one policy update for every two Q-function updates.

.. literalinclude:: ../../../../../genrl/agents/deep/td3/td3.py
    :lines: 95-121
    :lineno-start: 95

Training through the API
========================

.. code-block:: python

    from genrl.agents import TD3
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("MountainCarContinuous-v0")
    agent = TD3("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=4000)
    trainer.train()
    trainer.evaluate()
