Variational Inference
=====================

For an introduction to the Contextual Bandit problem, refer to :ref:`cb_overview`.

In this method, we try find a distribution
:math:`P_{\theta}(r | \mathbf{x}, a)` by minimising the KL divergence
with the true distribution. For the model we take a neueral network
where each weight is modelled by independant gaussians, also known as
Bayesian Neural Nets.

An example of using a variational inference based agent in ``genrl``
with bayesian net of hidden layer of 128 neurons and standard deviation
of 0.1 for al the weights -

.. code:: python

    from genrl.bandit import VariationalAgent, DCBTrainer

    agent = VariationalAgent(bandit, hidden_dims=[128], noise_std=0.1, device="cuda")

    trainer = DCBTrainer(agent, bandit)
    trainer.train()

Refer to the 
`VariationalAgent <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.variational>`__, 
and 
`DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__ 
docs for more details.
