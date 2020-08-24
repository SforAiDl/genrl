======================================
Deep Reinforcement Learning Background
======================================

Background
==========

The goal of Reinforcement Learning Algorithms is to maximize reward. This is usually achieved by having a policy :math:`\pi_{\theta}` perform optimal behavior. Let's denote this optimal policy by :math:`\pi_{\theta}^{*}`. For ease, we define the Reinforcement Learning problem as a Markov Decision Process. 

Markov Decision Process
=======================

An Markov Decision Process (MDP) is defined by :math:`(S, A, r, P_{a})` 
where,
 - :math:`S` is a set of States.
 - :math:`A` is a set of Actions.
 - :math:`r : S \rightarrow \mathbb{R}` is a reward function.
 - :math:`P_{a}(s, s')` is the transition probability that action :math:`a` in state :math:`s` leads to state :math:`s'`.

Often we define two functions, a policy function :math:`\pi_{\theta}(s,a)` and :math:`V_{\pi_{\theta}}(s)`.

Policy Function
===============

The policy is the agent's strategy, we our goal is to make it optimal. The optimal policy is usually denoted by :math:`\pi_{\theta}^{*}`. There are usually 2 types of policies:

Stochastic Policy
-----------------

The Policy Function is a stochastic variable defining a probability distribution over actions given states i.e. likelihood of every action when an agent is in a particular state. Formally,

.. math::
    \pi : S \times A \rightarrow [0,1]

.. math::
    a \sim \pi(a|s)


Deterministic Policy
--------------------
The Policy Function maps from States directly to Actions.

.. math::
    \pi : S \rightarrow A

.. math::
    a = \pi(s)

Value Function
==============

The Value Function is defined as the expected return obtained when we follow a policy :math:`\pi` starting from state S. Usually there are two types of value functions defined State Value Function and a State Action Value Function.

State Value Function
--------------------

The State Value Function is defined as the expected return starting from only State s.

.. math::
    V^{\pi}(s) = E\left[ R_{t} \right]

State Action Value Function
---------------------

The Action Value Function is defined as the expected return starting from a state s and a taking an action a.

.. math::
    Q^{\pi}(s,a) = E\left[ R_{t} \right] 

The Action Value Function is also known as the **Quality** Function as it would denote how good a particular action is for a state s.

Approximators
=============

Neural Networks are often used as approximators for Policy and Value Functions. In such a case, we say these are **parameterised** by :math:`\theta`. For e.g. :math:`\pi_{\theta}`.

Objective
=========

The objective is to choose/learn a policy that will maximize a cumulative function of rewards received at each step, typically the discounted reward over a potential infinite horizon. We formulate this cumulative function as 

.. math::

    E\left[{\sum_{t=0}^{\infty}{\gamma^{t} r_{t}}}\right]


where we choose an action according to our policy, :math:`a_{t} = \pi_{\theta}(s_{t})`. 
