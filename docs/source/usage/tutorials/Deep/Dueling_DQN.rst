======================
Dueling Deep Q-Network
======================

Objective
=========

The main objective of DQN is to learn a function approximator for the Q-function using a 
neural network. This is done by training the approximator to get as close to the Bellman Expectation of the Q-value function as possible by minimising the loss which is defined as:

.. math::

    E_{(s, a, s', r) \sim D}[r + \gamma max_{a'} Q(s', a';\theta_{i}^{-}) - Q(s, a; \theta_i)]^2

Dueling Deep Q-network modifies the architecture of a simple DQN into one better suited for model-free RL

Algorithm Details
=================

Network architechture
---------------------

The Dueling DQN architechture splits the single stream of fully connected layers in a normal DQN
into two separate streams : one representing the value function and the other representing the advantage function.
Advantage function.

.. math::

    A(s, a) = Q(s, a) - V(s, a)

The advantage for a state action pair represents how beneficial it is to take an action over others when in a particular state.
The dueling architechture can learn which states are or are not valuable without having to learn the effect of action for each state.
This is useful in instances when taking any action would affect the environment in any significant way.

Another layer combines the value stream and the advantage stream to get the Q-values

Combining the value and the advantage streams
---------------------------------------------

* Value Function : :math:`V(s; \theta, \beta)`
* Advantage Function : :math:`A(s, a; \theta, \alpha)`
    
where :math:`\theta` denotes the parameters of the underlying convolutional layers whereas :math:`\alpha` and :math:`\beta` are the parameters of the two separate streams of fully connected layers

The two stream cannot be simply added (:math:`Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha)`) to get the Q-values because:

* :math:`Q(s, a; \theta, \alpha, \beta)` is only a parameterized estimate of the true Q-function
* It would be wrong to assume that :math:`V(s; \theta, \beta)` and :math:`Q(s, a; \theta, \alpha)` are reasonable estimates of the value and the advantage functions

To address these concerns, we train in order to force the expected value of the advantage function to be zero (the expectation of advantage is always zero)

Thus, the combining module combines the value and advantage streams to get the Q-values in the following fashion:

.. math::

    Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + (A(s, a; \theta, \alpha) - max_{a'\in\mid A \mid}A(s, a'; \theta, \alpha))

Epsilon-Greedy Action Selection
-------------------------------

.. literalinclude:: ../../../../../genrl/agents/deep/dqn/dqn.py
    :lines: 97-129
    :lineno-start: 97

Similar to a normal DQN, the action exploration is stochastic wherein the greedy action is chosen with a probability of :math:`1 - \epsilon` and rest of the time, we sample the action randomly. During evaluation, we use only greedy actions to judge how well the agent performs.

Experience Replay
-----------------

Every transition occuring during the training is stored in a separate `Replay Buffer`

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 91-104
    :lineno-start: 91

The transitions are later sampled in batches from the replay buffer for updating the network

Update the Q Network
--------------------

Once enough number of transitions ae stored in the replay buffer, we start updating the Q-values according to the given objective. The loss function is defined in a fashion similar to a DQN. This allows
any new improvisations in training techniques of DQN such as Double DQN or NoisyNet DQN to be readily adapted in the dueling architechture.

.. literalinclude:: ../../../../../genrl/trainers/offpolicy.py
    :lines: 145-203
    :lineno-start: 145

Training through the API 
========================

.. code-block:: python

    from genrl.agents import DuelingDQN
    from genrl.environments import VectorEnv
    from genrl.trainers import OffPolicyTrainer

    env = VectorEnv("CartPole-v0")
    agent = DuelingDQN("mlp", env)
    trainer = OffpolicyTrainer(agent, env, max_timesteps=20000)
    trainer.train()
    trainer.evaluate()