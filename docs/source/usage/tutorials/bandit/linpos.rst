Linear Posterior Inference
==========================

For an introduction to the Contextual Bandit problem, refer to :ref:`cb_overview`.

In this agent we assume a linear relationship between context and reward
distribution of the form

.. math:: Y = X^T \beta + \epsilon \ \ \text{where} \ \epsilon \sim \mathcal{N}(0, \sigma^2)

We can utilise `bayesian linear
regression <https://en.wikipedia.org/wiki/Bayesian_linear_regression>`__
to find the parameters :math:`\beta` and :math:`\sigma`. Since our agent
is continually learning, the parameters of the model will being updated
according the (:math:`\mathbf{x}`, :math:`a`, :math:`r`) transitions it
observes.

For more complex non linear relations, we can make use of neural
networks to transform the context into a learned embedding space. The
above method can then be used on this latent embedding to model the
reward.

An example of using a neural network based linear posterior agent in
``genrl`` -

.. code:: python

    from genrl.bandit import NeuralLinearPosteriorAgent, DCBTrainer

    agent = NeuralLinearPosteriorAgent(bandit, lambda_prior=0.5, a0=2, b0=2, device="cuda")

    trainer = DCBTrainer(agent, bandit)
    trainer.train()

Note that the priors here are used to parameterise the initial
distribution over :math:`\beta` and :math:`\sigma`. More specificaly
``lambda_prior`` is used to parameterise a guassian distribution for
:math:`\beta` while ``a0`` and ``b0`` are paramters of an inverse gamma
distribution over :math:`\sigma^2`. These are updated over the course of
exploring a bandit. More details can be found in Section 3 of 
`this paper <https://arxiv.org/pdf/1802.09127.pdf>`__.

All hyperparameters can be tuned for individual use cases to improve
training efficiency and achieve convergence faster.

Refer to the 
`LinearPosteriorAgent <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.linpos>`__, 
`NeuralLinearPosteriorAgent <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.neural_linpos>`__ 
and 
`DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__ 
docs for more details.
