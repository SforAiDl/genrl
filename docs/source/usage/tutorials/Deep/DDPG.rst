===================================
Deep Deterministic Policy Gradients
===================================

Objective
=========

Deep Deterministic Policy Gradients (DDPG) is a model-free actor-critic algorithm which deals with continuous action spaces. One simple approach of dealing with continuous 
action spaces can be discretizing the action space. However, this gives rise to several problems, the most significant being that the size of the action-space increases exponentially 
with the number of degrees of freedom. DDPG builds up on *Deterministic Policy Gradients* to learn deterministic policies in high-dimensional continuous action-spaces.

Algorithms Details
==================

Deterministic Policy Gradient
-----------------------------

In cases with continuous action-spaces, using Q-learning like approach (greedy policy improvement) to learn deterministic policies is not feasible since it involves selecting the action with the maximum action value function 
at every step and it is not possible to check the action value for every possible action in case of continuous action spaces. 

.. math::
    \mu^{k+1}(s) = argmax_a Q^{\mu^{k}}(s, a)

This problem can be solved by considering the fact that a policy can be improved by moving it in the direction of increasing action-value function:

.. math::

    \nabla_\theta^{\mu} J = \mathbb{E_{s_t \sim \rho^{\beta}}}[\nabla_{\theta^{\mu^}}Q(s, a \vert \theta^{Q})\vert_{s=s_t, a=\mu(s_t \vert \theta^{\mu})}]
                          = \mathbb{E_{s_t \sim \rho^{\beta}}}[\nabla_a Q(s, a \vert \theta^{Q}) \vert_{s=s_t, a=\mu(s_t)}\nabla_{\theta_\mu}\mu(s \vert \theta^{\mu}) \vert_{s=s_t}]

Action Selection
----------------



