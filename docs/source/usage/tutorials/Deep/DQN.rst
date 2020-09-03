=====================
Deep Q-Networks (DQN)
=====================

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background<Background>`

Objective
=========

The DQN uses the concept of Q-learning. The objective is to get as close to the Bellman Expectation of the Q-value function as possible. This is done by minimising the loss function which is defined as

.. math::

    E_{(s, a, s', r) \sim D}[r + \gamma max_{a'} Q(s', a';\theta_{i}^{-}) - Q(s, a; \theta_i)]^2

Unlike in regular Q-learning, DQNs need more stability while updating so we often use a second neural network which we call our target model.

Algorithm Details
=================

Epsilon-Greedy Action Selection
-------------------------------

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/dqn.py
    :lines: 97-129
    :lineno-start: 97

We choose the greedy action with a probability of :math:`1 - \epsilon` and the rest of the time, we sample the action randomly. During evaluation, we use only greedy actions to judge how well the agent performs.

Experience Replay
-----------------

Whenever an experience is played through (during the training loop), the experience is stored in what we call a Replay Buffer.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

The transitions are later sampled in batches from the replay buffer for updating the network.

Update Q-value Network
----------------------

Once our Replay Buffer has enough experiences, we start updating the Q-value networks in the following code according to the above objective.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 145-203
    :lineno-start: 145

The function `get_q_values` calculates the Q-values of the states sampled from the replay buffer. The `get_target_q_values` function will get the Q-values of the same states as calculated by the target network.
The `update_params` function is used to calculate the MSE Loss between the Q-values and the Target Q-values and updated using Stochastic Gradient Descent.

Training through the API
========================

.. code-block:: python

    from genrl.agents import DQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("CartPole-v0")
    agent = DQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()

Variants of DQN
===============

Some of the other variants of DQN that we have implemented in the repo are:
- Double DQN 
- Dueling DQN
- Prioritized Replay DQN
- Noisy DQN
- Categorical DQN

For some extensions of the DQN (like DoubleDQN) we have provided the methods in a file under genrl/agents/dqn/utils.py

.. code-block:: python

    class DuelingDQN(DQN):
        def __init__(self, *args, **kwargs):
            super(DuelingDQN, self).__init__(*args, **kwargs)
            self.dqn_type = "dueling"  # You can choose "noisy" for NoisyDQN and "categorical" for CategoricalDQN
            self._create_model()

        def get_target_q_values(self, *args):
            return ddqn_q_target(self, *args)

The above two snippets define the same class. You can find similar APIs for the other variants in the `genrl/deep/agents/dqn` folder.

