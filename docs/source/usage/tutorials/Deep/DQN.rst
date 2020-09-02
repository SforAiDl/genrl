=====================
Deep Q-Networks (DQN)
=====================

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background<Background>`

Objective
=========

The DQN uses the concept of Q-learning. The objective is to get as close to the Bellman Expectation of the Q-value function as possible.

.. math::

    E_{s, a, s', r ~ D}[r + \gamma max_{a'} Q(s', a';\theta_{i}^{-}) - Q(s, a; \theta_i]^2

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

This Replay Buffer is later sampled from during updates (as you will see a little later)

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

.. code-block:: bash

    timestep         Episode          value_loss       epsilon          Episode Reward   
    24               0.0              0                0.9766           0                
    504              25.0             0                0.6154           19.44            
    1016             50.0             0.3241           0.2357           20.84            
    1442             75.0             0.5665           0.0972           13.92            
    6494             100.0            13.8086          0.0155           189.68           
    11076            125.0            47.9687          0.01             189.44           
    15910            150.0            18.478           0.01             191.36           
    20774            175.0            7.8271           0.01             197.04           
    Evaluated for 10 episodes, Mean Reward: 198.2, Std Deviation for the Reward: 3.627671429443411


Variants of DQN
===============

Some of the other variants of DQN that we have implemented in the repo are:
- Double DQN 
- Dueling DQN
- Prioritized Replay DQN
- Noisy DQN
- Categorical DQN

Double DQN
==========

The Double DQN takes the notion of Double Q-learning and applies it to the Q-network.

Objective
---------

.. math::

    E_{s, a, s', r ~ D}[r + \gamma Q(s', argmax_{a'} Q(s', a';\theta_{i}^{-})) - Q(s, a; \theta_i]^2

The only thing that differs with DoubleDQN is the `get_target_q_values` function as shown below.

.. code-block:: python
    
    from genrl.agents import DQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    class DoubleDQN(DQN):
        def __init__(self, *args, **kwargs):
            super(DoubleDQN, self).__init__(*args, **kwargs)
            self._create_model()

        def get_target_q_values(self, next_states, rewards, dones):
            next_q_value_dist = self.model(next_states)
            next_best_actions = torch.argmax(next_q_value_dist, dim=-1).unsqueeze(-1)

            rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

            next_q_target_value_dist = self.target_model(next_states)
            max_next_q_target_values = next_q_target_value_dist.gather(2, next_best_actions)
            target_q_values = rewards + agent.gamma * torch.mul(
                max_next_q_target_values, (1 - dones)
            )
            return target_q_values

    env = VectorEnv("CartPole-v0")
    agent = DoubleDQN("mlp", env)
    trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()
    

.. code-block:: bash
    timestep         Episode          value_loss       epsilon          Episode Reward   
    24               0.0              0                0.9766           0                
    720              25.0             0                0.5184           26.96            
    1168             50.0             0.49             0.1646           18.6             
    3248             75.0             4.1546           0.0326           74.88            
    7512             100.0            7.3164           0.0102           166.36           
    12424            125.0            12.3175          0.01             200.0            
    Evaluated for 10 episodes, Mean Reward: 200.0, Std Deviation for the Reward: 0.0

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
