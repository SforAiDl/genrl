==================================
Multi Agent Advantage Actor Critic
==================================

Please visit the following tutorials before you run by this one:

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background<Background>`

For background on Advantage Actor Critic, :ref:`Deep RL A2C<A2C>`

Motivation
==========

- Multiagents interact in common environment, each agent with it's own sensor, effectors, goals ..
- Agents have to either coordinate or compete with each other in order to achieve their goals

Objective
=========

1. To choose/learn a policy that will increase the probability of landing an action that has higher expected return of the multiagent sysetem, in a cooperative or competitive setting.

Environment
===========
In this tutorial, we will see how a single agent algorithm can be extended to a multiagent setting. 

Environment: Simple Spread 

Details: N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions.

Algorithm Details
=================

Network details
---------------------------
We use a shared Actor-Critic network as it is a fully observable environment, the input space can be shared by both actor and critic.

.. literalinclude:: ../../../../../../contrib/maa2c/a2c.py
   :lines: 7-61
   :lineno-start: 7


Action Selection and Values
---------------------------
Here we need to pass the states of every agent to the network to one at a time. We adopt a decentralized approach to train every agent in order to achieve cooperation.

Note: We sample a **stochastic action** from the distribution on the action space and provide ``False`` as an argument to ``select_action`` as we want the action number not in the form of a one-hot encoding.

.. literalinclude:: ../../../../../../contrib/maa2c/a2c_agent.py
   :lines: 62-67
   :lineno-start: 62

Here we need to pass the states of every agent to the network to 

.. literalinclude:: ../../../../../../contrib/maa2c/a2c.py
   :lines: 34-44
   :lineno-start: 34

Collect Experience
------------------

To make our agent learn, we first need to collect some experience in an online fashion. For this we make use of the ``collect_rollouts`` method. It simply collects states, next-states, actions, rewards, dones and stores it in a list.

.. literalinclude:: ../../../../../../contrib/maa2c/a2c_agent.py
   :lines: 69-110
   :lineno-start: 69

For updation, we would need to compute advantages, discounted rewards and losses from this experience. So, we store our experience in a Buffer (list in our case). 

This method returns the Q-values and dones.

Compute discounted Returns, Advantages and Losses
-------------------------------------------------

Next we can compute the advantages and the actual discounted returns for each state. This is further used to calculate ``value_loss`` ``policy_loss`` ``entropy`` and ``total_loss``. This can be done very easily by simply calling ``get_traj_loss``. 

.. literalinclude:: ../../../../../../contrib/maa2c/a2c_agent.py
   :lines: 112-148
   :lineno-start: 112

Update Equations
----------------

The following losses are used to update the network parameters.

.. literalinclude:: ../../../../../../contrib/maa2c/a2c_agent.py
   :lines: 112-148
   :lineno-start: 112

Plotting graphs
---------------

We call ``get_logging_params`` method to plot tensorboard plots.

.. literalinclude:: ../../../../../../contrib/maa2c/a2c_agent.py
   :lines: 158-168
   :lineno-start: 158

There is a separate script in order to go through the entire training process:

.. literalinclude:: ../../../../../../contrib/maa2c/examples/maa2c.py

Training through the API
========================

Firstly you will need to clone the multiagent-particle-env from: https://github.com/AdityaKapoor74/multiagent-particle-envs in the examples directory and follow the instructions in order to setup the environment.

.. code-block:: python

    import sys
    sys.path.append("add path to genrl/genrl in case you get import error") 
    import numpy as np

    from contrib.maa2c.a2c_agent import A2CAgent
    from simple_spread_test import make_env

    from utils import TensorboardLogger


    def train(max_episodes, timesteps_per_eps, logdir):

      tensorboard_logger = TensorboardLogger(logdir)

      environment = make_env(scenario_name="simple_spread",benchmark=False)

      a2c_agent = A2CAgent(environment, 2e-4, 0.99, None, 0.008, timesteps_per_eps)

      for episode in range(1,max_episodes+1):

        print("EPISODE",episode)

        states = np.asarray(environment.reset())

        values, dones = a2c_agent.collect_rollouts(states)

        a2c_agent.get_traj_loss(values,dones)

        a2c_agent.update_policy()

        params = a2c_agent.get_logging_params()
        params["episode"] = episode

        tensorboard_logger.write(params)



      tensorboard_logger.close()





    if __name__ == '__main__':

      train(max_episodes = 10000, timesteps_per_eps = 300, logdir="./runs/")
