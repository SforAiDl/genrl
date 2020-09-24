=====================
Double Deep Q-Network
=====================

Objective
=========

Double DQN builds upon the notion of Double Q-Learning and extends it to Deep Q-networks. We use function approximators for predicting the Q-values of the states and a function approximator is always corrupted with some noise.
Now, when we maximise over the values of state-action pairs while calculating the target for the TD-update, the maximum is taken over the true values plus the noise. Thus, the maximum of a noisy function is always bigger than the maximum 
of the true function:

.. math::

    E[max(X_1, X_2)] \ge max[E(X_1), E(X_2)]
    
where :math:`X_1` and :math:`X_2` are two random variables. This leads to overestimations of the values of state-action pairs and cnsequently suboptimal action selection. This overestimation is bound to propagate and increase over the course of 
multiple updates because the same approximator is used to select the maximum action and to estimate it's Q-value.

.. math::

    max_{a'}Q_{\phi'}(s', a') = Q_{\phi'}(s', argmax_{a'}Q_{\phi'}(s', a'))
    
This problem can be solved by decoupling the action selection and the value estimation using two separate function approximators(and hence different noise distributions) for both the purposes which is what a Double-DQN does. The loss function is defined as:

.. math::

    E_{s, a \sim \rho(.)}[(y^{DoubleDQN} - Q(s, a; \theta))^{2}]

Algorithm Details
=================

Epsilon-Greedy Action Selection
-------------------------------

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/dqn.py
    :lines: 97-129
    :lineno-start: 97

The action exploration is stochastic wherein the greedy action is chosen with a probability of :math:`1 - \epsilon` and rest of the time, we sample the action randomly. During evaluation, we use only greedy actions to judge how well the agent performs.

Experience Replay
-----------------

Every transition occuring during the training is stored in a separate `Replay Buffer`

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

The transitions are later sampled in batches from the replay buffer for updating the network.

Update the Q-Network
--------------------

Doble DQN decouples the selection of the action from the evaluation of the Q-values while calculating the target value for the update. The loss function for a time step t is defined as:

.. math::

    L_t(\theta_t) = E_{s, a \sim \rho(.)}[(y_{t}^{DoubleDQN} - Q(s, a; \theta_t))^{2}]
    
    y_t^{DoubleDQN} = R_{t+1} + \gamma Q(s_{t+1}, argmax_{a} Q(s_{t+1}, a; \theta_{t}), \theta_t^{-})

The only thing that differs with DoubleDQN is the `get_target_q_values` function as shown below.

.. code-block:: python
    
    from genrl.agents import DQN
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

Training through the API
========================

.. code-block:: python

    from genrl.agents import DoubleDQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

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