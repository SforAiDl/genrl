# Linear Posterior Inference

Here, we assume a linear relationship between context and reward distribution of the form

$$Y = X^T \beta + \epsilon \ \ \text{where} \ \epsilon \sim \mathcal{N}(0, \sigma^2)$$

We can utilise [bayesian linear regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression) to find the parameters $\beta$ and $\sigma$. Since our agent is continually learning, the parameters of the model will being updated according the ($\mathbf{x}$, $a$, $r$) transitions it observes.

For more complex non linear relations, we can make use of neural networks to transform the context into a learned embedding space. The above method can then be used on this latent embedding to model the reward.

An example of using a neural network based linear posterior agent in `genrl` -

```python
from genrl.bandit import NeuralLinearPosteriorAgent, DCBTrainer

agent = NeuralLinearPosteriorAgent(bandit, lambda_prior=0.5, a0=2, b0=2, device="cuda")

trainer = DCBTrainer(agent, bandit)
trainer.train()
```

Note that the priors here are used to parameterise the initial distribution over $\beta$ and $\sigma$. More specificaly `lambda_prior` is used to parameterise a guassian distribution for $\beta$ while `a0` and `b0` are paramters of an inverse gamma distribution over $\sigma^2$. These are updated over the course of exploring a bandit. More details can be found in Section 3 of this [paper](https://arxiv.org/pdf/1802.09127.pdf).

All hyperparameters can be tuned for individual use cases to improve training efficiency and achieve convergence faster.

Refer to the `LinearPosteriorAgent` and `NeuralLinearPosteriorAgent` docs for more information.