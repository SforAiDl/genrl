.. GenRL documentation master file, created by
   sphinx-quickstart on Mon Jun 29 23:38:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GenRL's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   tutorials/index

.. Features 
.. ========
.. * Unified Trainer and Logging class: code reusability and high-level UI
.. * Ready-made algorithm implementations: ready-made implementations of popular RL algorithms.
.. * Extensive Benchmarking
.. * Environment implementations
.. * Heavy Encapsulation useful for new algorithms

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
            
..       3. :mod:`genrl.classical.SARSA`: SARSA
..       4. :mod:`genrl.classical.QLearning`: Q-Learning
