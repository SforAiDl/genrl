Parameter Noise Sampling
========================

For an introduction to the Contextual Bandit problem, refer to :ref:`cb_overview`.

One of the ways to improve exploration of our algorithms is to introduce
noise into the weights of the neural network while selecting actions.
This does not affect the gradients but will have a similar effect to
epsilon greedy exploration.

The noise distribution is regularly updated during training to keep the
KL divergence of the prediction and noise predictions within certain
limits.

An example of using a noise sampling based agent in ``genrl`` with noise
standard deviation as 0.1, KL divergence limit as 0.1 and batch size for
updating the noise distribution as 128 -

.. code:: python

    from genrl.bandit import BootstrapNeuralAgent, DCBTrainer

    agent = NeuralNoiseSamplingAgent(bandit, hidden_dims=[128], noise_std_dev=0.1, eps=0.1, noise_update_batch_size=128, device="cuda")

    trainer = DCBTrainer(agent, bandit)
    trainer.train()

Refer to the 
`NeuralNoiseSamplingAgent <../../../api/bandit/genrl.agents.bandits.contextual.html#module-genrl.agents.bandits.contextual.neural_noise>`__, 
and 
`DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__ 
docs for more details.
