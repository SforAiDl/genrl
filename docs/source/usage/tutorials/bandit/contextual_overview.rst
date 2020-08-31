.. _cb_overview:

Contextual Bandits Overview
===========================

Problem Setting
---------------

To get some background on the basic multi armed bandit problem, we recommend
that you go through the :ref:`bandit_overview` first. The contextual bandit 
(CB) problem varies from the basic case in that at each timestep, a context 
vector :math:`x \in \mathbb{R}^d` is presented to the agent. The agent must
then decide on an action :math:`a \in \mathcal{A}` to take based on that
context. After the action is taken, the reward :math:`r \in \mathbb{R}`
for only that action is revealed to the agent (a feature of all
reinforcement learning problems). The aim of the agent remains the same
- minimising regret and thus finding an optimal policy.

Here you still have the problem of exploration vs exploitation, but the
agent also needs to find some relation between the context and reward.

A Simple Example
------------------

Lets consider the simplest case of the CB problem. Instead of having
only one :math:`k`-armed bandit that needs to be solved, say we have
:math:`m` different :math:`k`-armed Bernoulli bandits. At each timestep,
the context presented is the number of the bandit for which an action
needs to be selected: :math:`i \in \mathbb{I}` where :math:`0 < i \le m`

Although real life CB problems usually have much higher dimensional
contexts, such a toy problem can be usefull for testing and debugging
agents.

To instantiate a Bernoulli bandit with :math:`m =10` and :math:`k = 5`
(10 different 5-armed bandits) -

.. code:: python

    from genrl.bandit import BernoulliMAB

    bandit = BernoulliMAB(bandits=10, arms=5, context_type="int")

Note that this is using the same ``BernoulliMAB`` as in the simple
bandit case except that instead of the ``bandits`` argument defaulting
to ``1``, we are explicitly saying we want multiple bandits (a
contexutal case)

Suppose you want to solve this bandit with a UCB based policy.

.. code:: python

    from genrl.bandit import UCBMABAgent

    agent = UCBMABAgent(bandit)
    context = bandit.reset()

    action = agent.select_action(context)
    new_context, reward = bandit.step(action)

To train the agent, you an set up a loop which calls the
``update_params`` method on the agent whenever you want to agent to
learn from actions it has taken. For convinience it is highly
recommended to use the ``MABTrainer`` in such cases.

Data based Conextual Bandits
----------------------------

Lets consider a more realistic class of CB problem. I real life, you the
CB setting is usually used to model recommendation or classification
problems. Here, instead of getting an integer as the context, you will
get a :math:`d`-dimensional feature vector
:math:`\mathbf{x} \in \mathbb{R}^d`. This is also different from regular
classification since you only get the reward :math:`r \in \mathbb{R}`
for the action you have taken.

While tabular solutions can work well for integer contexts (see the
implentation of any ``genrl.bandit.MABAgent`` for details), when you
have a high dimensional vector, the agent should be able to infer the
complex relation between the contexts and rewards. This can be done by
modelling a conditional distribution over rewards for each action given
the context.

.. math::  P(r | a, \mathbf{x})

There are many ways to do this. For a detailed explanation and
comparison of contextual bandit methods you can refer to
`this paper <https://arxiv.org/pdf/1802.09127.pdf>`__.

The following are the agents implemented in ``genrl``

-  `Linear Posterior Inference <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.inpos>`__
-  `Neural Network based Linear <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.neural_linpos>`__
-  `Variational <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.variational>`__
-  `Neural Netowork based Espilon Greedy <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.neural_greedy>`__
-  `Bootstrap <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.bootstrap_neural>`__
-  `Parameter noise Sampling <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.neural_noise_sampling>`__

You can find the tutorials for most of these in :ref:`bandit_tutorials`.

All the methods which use neural networks, provide an option to train
and evaluate with dropout, have a decaying learning rate and a limit for
gradient clipping. The sizes of hidden layers for the networks can also
be specified. Refer to docs of the specific agents to see how to use
these options.

Individual agents will have other method specific paramters to control
behavior. Although default values have been provided, it may be
neccessary to tune these for individual use cases.

The following bandits based on datasets are implemented in ``genrl``

-  `Adult Census Income Dataset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.adult_bandit>`__
-  `US Census Dataset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.census_bandit>`__
-  `Forest covertype Datset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.covertype_bandit>`__
-  `MAGIC Gamma Telescope dataset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.magic_bandit>`__
-  `Mushroom Dataset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.mushroom_bandit>`__
-  `Statlog Space Shuttle Dataset <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.statlog_bandit>`__

For each bandit, while instatiating an object you can either specify a
path to the data file or pass ``download=True`` as an argument to
download the data directly.

Data based Bandit Example
-------------------------

For this example, we'll model the
`Statlog <https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)>`__
dataset as a bandit problem. You can read more about the bandit in the
`Statlog docs <../../../api/bandit/genrl.utils.data_bandits.html#module-genrl.utils.data_bandits.statlog_bandit>`__.
In brief we have the number of arms as :math:`k = 7` and
dimension of context vector as :math:`d = 9`. The agent will get a
reward :math:`r =1` if it selects the correct arm else :math:`r = 0`.

.. code:: python

    from genrl.bandit import StatlogDataBandit

    bandit = StatlogDataBandit(download=True)
    context = bandit.reset()

Suppose you want to solve this bandit with a Greedy neural network based
policy.

.. code:: python

    from genrl.bandit import NeuralLinearPosteriorAgent

    agent = NeuralLinearPosteriorAgent(bandit)
    context = bandit.reset()

    action = agent.select_action(context)
    new_context, reward = bandit.step(action)

To train the agent, we highly reccomend using the ``DCBTrainer``. You
can refer to the implementation of the ``train`` function to get an idea
of how to implemente your own training loop.

.. code:: python

    from genrl.bandit import DCBTrainer

    trainer = DCBTrainer(agent, bandit)
    trainer.train(timesteps=5000, batch_size=32)


Further material about bandits
------------------------------
1. `Deep Contextual Multi-armed Bandits <https://arxiv.org/pdf/1807.09809.pdf>`__, Collier and Llorens, 2018
2. `Deep Bayesian Bandits Showdown <https://arxiv.org/pdf/1802.09127.pdf>`__, Riquelme∗ et al, 2018
3. `A Contextual Bandit Bake-off <https://arxiv.org/pdf/1802.09127.pdf>`__, Bietti et al, 2020
