=======================
Vanilla Policy Gradient
=======================

For background on Deep RL, its core definitions and problem formulations refer to :ref:`Deep RL Background`

Objective
=========

The objective is to choose/learn a policy that will maximize a cumulative function of rewards received at each step, typically the discounted reward over a potential infinite horizon. We formulate this cumulative function as 

.. math::

    E\left[{\sum_{t=0}^{\infty}{\gamma^{t} r_{t}}}\right]


where we choose the action :math:`a_{t} = \pi_{\theta}(s_{t})`. 

Algorithm Details
=================

Collect Experience
------------------

To make our agent learn, we first need to collect some experience in an online fashion. For this we make use of the ``collect_rollouts`` method. This method is defined in the ``OnPolicyAgent`` Base Class. 

.. literalinclude:: ../../../../../genrl/deep/agents/base.py
   :lines: 141-163
   :lineno-start: 141

For updation, we would need to compute advantages from this experience. So, we store our experience in a Rollout Buffer. 
Action Selection
----------------

.. literalinclude:: ../../../../../genrl/deep/agents/vpg/vpg.py
   :lines: 99-115
   :lineno-start: 99

Note: We sample a **stochastic action** from the distribution on the action space by providing ``False`` as an argument to ``select_action``.

For practical purposes we would assume that we are working with a finite horizon MDP.

Update Equations
----------------

Let :math:`\pi_{\theta}` denote a policy with parameters :math:`\theta`, and :math:`J(\pi_{\theta})` denote the expected finite-horizon undiscounted return of the policy. 

At each update timestep, we get value and log probabilities:

.. literalinclude:: ../../../../../genrl/deep/agents/vpg/vpg.py
   :lines: 123-126
   :lineno-start: 123

Now, that we have the log probabilities we calculate the gradient of :math:`J(\pi_{\theta})` as:

.. math:: 
    
    \nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}\left[{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)
        }\right],

where :math:`\tau` is the trajectory.

.. literalinclude:: ../../../../../genrl/deep/agents/vpg/vpg.py
   :lines: 134-145
   :lineno-start: 134

We then update the policy parameters via stochastic gradient ascent:

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

.. literalinclude:: ../../../../../genrl/deep/agents/vpg/vpg.py
   :lines: 148-151
   :lineno-start: 148

The key idea underlying vanilla policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.

Training through the API
========================

.. code-block:: python

    import gym

    from genrl import VPG
    from genrl.deep.common import OnPolicyTrainer
    from genrl.environments import VectorEnv

    env = VectorEnv("CartPole-v0")
    agent = VPG('mlp', env)
    trainer = OnPolicyTrainer(agent, env, log_mode=['stdout'])
    trainer.train()

.. code-block:: bash

    timestep         Episode          loss             mean_reward      
    0                0                9.1853           22.3825          
    20480            10               24.5517          80.3137          
    40960            20               24.4992          117.7011         
    61440            30               22.578           121.543          
    81920            40               20.423           114.7339         
    102400           50               21.7225          128.4013         
    122880           60               21.0566          116.034          
    143360           70               21.628           115.0562         
    163840           80               23.1384          133.4202         
    184320           90               23.2824          133.4202         
    204800           100              26.3477          147.87           
    225280           110              26.7198          139.7952         
    245760           120              30.0402          184.5045         
    266240           130              30.293           178.8646         
    286720           140              29.4063          162.5397         
    307200           150              30.9759          183.6771         
    327680           160              30.6517          186.1818         
    348160           170              31.7742          184.5045         
    368640           180              30.4608          186.1818         
    389120           190              30.2635          186.1818     
