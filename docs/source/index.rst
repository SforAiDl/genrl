.. GenRL documentation master file, created by
   sphinx-quickstart on Mon Jun 29 23:38:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GenRL's documentation!
=================================

Features 
========
* Unified Trainer and Logging class: code reusability and high-level UI
* Ready-made algorithm implementations: ready-made implementations of popular RL algorithms.
* Extensive Benchmarking
* Environment implementations
* Heavy Encapsulation useful for new algorithms

Contents
========
.. toctree::
   :maxdepth: 2
   
   install
   getting_started
   algorithms/index

.. Algorithms
.. ===========
.. GenRL currently supports the following algorithms:

.. * Deep

..       1. :mod:`genrl.VPG`: Vanilla Policy Gradients 
..       2. :mod:`genrl.A2C`: Advantage Actor-Critic
..       3. :mod:`genrl.PPO1`: Proximal Policy Optimization
..       4. :mod:`genrl.DQN`: Deep Q-Learning
..       5. :mod:`genrl.DDPG`: Deep Deterministic Policy Gradients
..       6. :mod:`genrl.TD3`: Twin Delayed Deep Determinsitic Policy Gradients
..       7. :mod:`genrl.SAC`: Soft Actor-Critic
   
.. * Classical

..       1. :mod:`genrl.classical.bandits`: Multi-armed Bandits

..             + :mod:`genrl.EpsGreedyPolicy`: Eps-Greedy
..             + :mod:`genrl.UCBPolicy`: Upper Confidence Bound
..             + :mod:`genrl.BayesianPolicy`: Bayesian Bandit
..             + :mod:`genrl.ThompsonSamplingPolicy`: Thompson Sampling
..             + :mod:`genrl.GradientPolicy`: Softmax Explorer
            
..       2. Contextual Bandits

..             + :mod:`genrl.EpsGreedyCBPolicy`: Eps-Greedy
..             + :mod:`genrl.UCBCBPolicy`: Upper Confidence Bound
..             + :mod:`genrl.BayesianCBPolicy`: Bayesian Bandit
..             + :mod:`genrl.ThompsonSamplingCBPolicy`: Thompson Sampling
..             + :mod:`genrl.GradientCBPolicy`: Softmax Explorer
            

Train Vanilla Policy Gradient on Vectorized CartPole-v1

.. code-block:: python

      from genrl import VPG
      from genrl.deep.common import OnPolicyTrainer
      from genrl.environments import VectorEnv

      # Specify some hyperparameters
      n_envs = 10
      epochs = 15
      eval_episodes = 10
      arch = "mlp"
      log = ["stdout,tensorboard"] # Specify logging type as a comma-separated list
      
      # Initialize Agent and Environment
      env = VectorEnv("CartPole-v1", n_envs)
      agent = VPG("mlp", env)

      # Trainer
      trainer = OnPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
      trainer.train()

      # Evaluation
      trainer.render = True
      trainer.evaluate()

Train Proximal Policy Optimization (PPO) on Vectorized LunarLander-v2

.. code-block:: python

      from genrl import PPO1
      from genrl.deep.common import OnPolicyTrainer
      from genrl.environments import VectorEnv

      # Specify some hyperparameters
      n_envs = 10
      epochs = 40
      eval_episodes = 20
      arch = "mlp"
      log = ["stdout,tensorboard"] # Specify logging type as a comma-separated list
      
      # Initialize Agent and Environment
      env = VectorEnv("CartPole-v1", n_envs)
      agent = PPO1("mlp", env)

      # Trainer
      trainer = OnPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
      trainer.train()

      # Evaluation
      trainer.render = True
      trainer.evaluate()

Train Soft Actor-Critic (SAC) on Vectorized Pendulum-v0

.. code-block:: python

      from genrl import PPO1
      from genrl.deep.common import OffPolicyTrainer
      from genrl.environments import VectorEnv

      # Specify some hyperparameters
      n_envs = 10
      epochs = 40
      eval_episodes = 20
      arch = "mlp"
      log = ["stdout,tensorboard"] # Specify logging type as a comma-separated list
      
      # Initialize Agent and Environment
      env = VectorEnv("Pendulum-v0", n_envs)
      agent = SAC("mlp", env)

      # Trainer
      trainer = OffPolicyTrainer(agent, env, log, epochs = epochs, evaluate_episodes = eval_episodes)
      trainer.train()

      # Evaluation
      trainer.render = True
      trainer.evaluate()
=======
..       3. :mod:`genrl.classical.SARSA`: SARSA
..       4. :mod:`genrl.classical.QLearning`: Q-Learning
