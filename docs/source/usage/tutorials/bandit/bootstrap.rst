Bootstrap
=========

For an introduction to the Contextual Bandit problem, refer to :ref:`cb_overview`.

In the bootstrap agent multiple different neural network based models are trained
simultaneously. Different transition databases are maintained for each
model and every time we observe a transition it is added to each dataset
with some probability. At each timestep, the model used to select an
action is chosen randomly from the set of models.

By having multiple different models initialised with different random
weights, we promote the exploration of the loss landscape which may have
multiple different local optima.

An example of using a bootstrap based agent in ``genrl`` with 10 models
with a hidden layer of 128 neurons which also uses dropout for training
-

.. code:: python

    from genrl.bandit import BootstrapNeuralAgent, DCBTrainer

    agent = BootstrapNeuralAgent(bandit, hidden_dims=[128], n=10, dropout_p=0.5, device="cuda")

    trainer = DCBTrainer(agent, bandit)
    trainer.train()

Refer to the 
`BootstrapNeuralAgent <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.bootstrap_neural>`__ 
and 
`DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__ 
docs for more details.
