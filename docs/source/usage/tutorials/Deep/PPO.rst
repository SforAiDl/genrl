============================
Proximal Policy Optimization
============================

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background<Background>`

Objective
=========

The objective is to maximize the discounted cumulative reward function:

.. math::

    E\left[{\sum_{t=0}^{\infty}{\gamma^{t} r_{t}}}\right]

The Proximal Policy Optimization Algorithm is very similar to the Advantage Actor Critic Algorithm except we add multiply the advantages with a ratio between the log probability of actions at experience collection time and at updation time. What this does is - helps in establishing a trust region for not moving too away from the old policy and at the same time taking gradient ascent steps in the directions of actions which result in positive advantages.


where we choose the action :math:`a_{t} = \pi_{\theta}(s_{t})`. 

Algorithm Details
=================

Action Selection and Values
---------------------------

.. literalinclude:: ../../../../../genrl/deep/agents/ppo1/ppo1.py
   :lines: 119-125
   :lineno-start: 119

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

.. literalinclude:: ../../../../../genrl/deep/agents/ppo1/ppo1.py
   :lines: 127-135
   :lineno-start: 127

In the case of PPO our loss function is:

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
    \right),

where :math:`\tau` is the trajectory.

.. literalinclude:: ../../../../../genrl/deep/agents/ppo1/ppo1.py
   :lines: 143-173
   :lineno-start: 143

We then update the policy parameters via stochastic gradient ascent:

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

.. literalinclude:: ../../../../../genrl/deep/agents/ppo1/ppo1.py
   :lines: 175-183
   :lineno-start: 175

Training through the API
========================

.. code-block:: python

    import gym

    from genrl import PPO1
    from genrl.deep.common import OnPolicyTrainer
    from genrl.environments import VectorEnv

    env = VectorEnv("CartPole-v0")
    agent = PPO1('mlp', env)
    trainer = OnPolicyTrainer(agent, env, log_mode=['stdout'])
    trainer.train()
