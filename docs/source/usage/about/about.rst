=====
About
=====

Introduction
============

Reinforcement Learning has taken massive leaps forward in extending current AI research. David Silver's paper on playing Atari with Deep Reinforcement Learning can be considered one of the seminal papers in establishing a completely new landscape of Reinforcement Learning Research. With applications in Robotics, Healthcare and numerous other domains, RL has become the prime mechanism of modelling Sequential Decision Making through AI. 

Yet, current libraries and resources in Reinforcement Learning are either very limited, messy and/or are scattered. OpenAI's Spinning Up is a great resource for getting started with Deep Reinforcement Learning but it fails to cover more basic concepts in Reinforcement Learning for e.g. Multi Armed Bandits. garage is a great resource for reproducing and evaluating RL algorithms but it fails to introduce a newbie to RL.

With GenRL, our goal is three-fold:
- To educate the user about Reinforcement learning.
- Easy to understand implementations of State of the Art Reinforcement Learning Algorithms.
- Providing utilities for developing and evaluating new RL algorithms. Or in a sense be able to implement any new RL algorithm in less than 200 lines.

Policies and Values
===================
Modern research on Reinforcement Learning is majorly based on Markov Decision Processes. Policy and Value Functions are one of the core parts of such a problem formulation. And so, polices and values form one of the core parts of our library.

Trainers and Loggers
====================

Trainers
--------

Most current algorithms follow a standard procedure of training. Considering a classification between On-Policy and Off-Policy Algorithms, we provide high level APIs through Trainers which can be coupled with Agents and Environments for training seamlessly.

Lets take the example of an On-Policy Algorithm, Proximal Policy Optimization. In our Agent, we make sure to define three methods: ``collect_rollouts``, ``get_traj_loss`` and finally ``update_policy``. 

.. literalinclude:: ../../../../genrl/deep/common/trainer.py
   :lines: 507-511
   :lineno-start: 507

The ``OnPolicyTrainer`` simply calls these functions and enables high level usage by simple defining of three methods.

Loggers
-------

At the moment, we support three different types of Loggers. ``HumanOutputFormat``, ``TensorboardLogger`` and ``CSVLogger``. Any of these loggers can be initialized really easily by the top level ``Logger`` class and specifying the individual formats in which logging should performed.

.. code-block:: python

    logger = Logger(logdir='logs/', formats=['stdout', 'tensorboard'])

After which logger can perform logging easily by providing it with dictionaries of data. For e.g.

.. code-block:: python

    logger.write({"logger":0})

Note: The Tensorboard logger requires an extra x-axis parameter, as it plots data rather than just show it in a tabular format.

Agent Encapsulation
===================

WIP

Environments
============
Wrappers
