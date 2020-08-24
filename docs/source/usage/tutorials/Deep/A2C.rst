======================
Advantage Actor Critic
======================

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background<Background>`

Objective
=========

The objective is to maximize the discounted cumulative reward function:

.. math::

    E\left[{\sum_{t=0}^{\infty}{\gamma^{t} r_{t}}}\right]

This comprises of two parts in the Adantage Actor Critic Algorithm:

1. To choose/learn a policy that will increase the probability of landing an action that has higher expected return than the value of just the state and decrease the probability of landing an action that has lower expected return than the value of the state. The Advantage is computed as:

.. math::
    A(s,a) = Q(s,a) - V(s)

2. To learn a State Action Value Function (in the name of **Critic**) that estimates the future cumulative rewards given the current state and action. This function helps the policy in evaluation potential state, action pairs.


where we choose the action :math:`a_{t} = \pi_{\theta}(s_{t})`. 

Algorithm Details
=================

Action Selection and Values
---------------------------

.. literalinclude:: ../../../../../genrl/deep/agents/a2c/a2c.py
   :lines: 130-149
   :lineno-start: 130

``ac`` here is an object of the ``ActorCritic`` class, which defined two methods: ``get_value`` and ``get_action`` and ofcourse they return the value approximation from the Critic and action from the Actor.

Note: We sample a **stochastic action** from the distribution on the action space by providing ``False`` as an argument to ``select_action``.

For practical purposes we would assume that we are working with a finite horizon MDP.

Collect Experience
------------------

To make our agent learn, we first need to collect some experience in an online fashion. For this we make use of the ``collect_rollouts`` method. This method is defined in the ``OnPolicyAgent`` Base Class. 

.. literalinclude:: ../../../../../genrl/deep/agents/base.py
   :lines: 141-163
   :lineno-start: 141

For updation, we would need to compute advantages from this experience. So, we store our experience in a Rollout Buffer. 

Compute discounted Returns and Advantages
-----------------------------------------

Next we can compute the advantages and the actual discounted returns for each state. This can be done very easily by simply calling ``compute_returns_and_advantage``. Note this implementation of the rollout buffer is borrowed from Stable Baselines.

.. literalinclude:: ../../../../../genrl/deep/common/rollout_storage.py
   :lines: 222-238
   :lineno-start: 222


Update Equations
----------------

Let :math:`\pi_{\theta}` denote a policy with parameters :math:`\theta`, and :math:`J(\pi_{\theta})` denote the expected finite-horizon undiscounted return of the policy. 

At each update timestep, we get value and log probabilities:

.. literalinclude:: ../../../../../genrl/deep/agents/a2c/a2c.py
   :lines: 158-162
   :lineno-start: 158

Now, that we have the log probabilities we calculate the gradient of :math:`J(\pi_{\theta})` as:

.. math:: 
    
    \nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}\left[{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        }\right],

where :math:`\tau` is the trajectory.

.. literalinclude:: ../../../../../genrl/deep/agents/a2c/a2c.py
   :lines: 165-183
   :lineno-start: 165

We then update the policy parameters via stochastic gradient ascent:

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

.. literalinclude:: ../../../../../genrl/deep/agents/vpg/vpg.py
   :lines: 185-193
   :lineno-start: 185

The key idea underlying Advantage Actor Critic Algorithm is to push up the probabilities of actions that lead to higher return than the expected return of that state, and push down the probabilities of actions that lead to lower return than the expected return of that state, until you arrive at the optimal policy.

Training through the API
========================

.. code-block:: python

    import gym

    from genrl import A2C
    from genrl.deep.common import OnPolicyTrainer
    from genrl.environments import VectorEnv

    env = VectorEnv("CartPole-v0")
    agent = A2C('mlp', env)
    trainer = OnPolicyTrainer(agent, env, log_mode=['stdout'])
    trainer.train()


    import sys
    sys.path.append("/home/aditya/Desktop/genrl/genrl") #add path
    import numpy as np

    from contrib.maa2c.a2c_agent import A2CAgent
    from simple_spread_test import make_env
    # from genrl.deep.common.trainer import OnPolicyTrainer
    # from genrl.environments import VectorEnv

    from utils import TensorboardLogger
    # from genrl.deep.common.logger import TensorboardLogger


    def train(max_episodes, timesteps_per_eps, logdir):

      # CUSTOM
      tensorboard_logger = TensorboardLogger(logdir)
      # vec_env = VectorEnv(
     #        env_id="simple_spread", n_envs=1, parallel=False, env_type="multi"
     #        )

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

      # on_policy_trainer = OnPolicyTrainer(agent=a2c_agent,env=vec_env, log_mode=["tensorboard"], log_key="Episode" , save_interval=100, save_model="checkpoints", steps_per_epoch=timesteps_per_eps, epochs=max_episodes,
      #   log_interval=1, batch_size=1)

      # on_policy_trainer.train()





    if __name__ == '__main__':

      train(max_episodes = 10000, timesteps_per_eps = 300, logdir="./runs/")

