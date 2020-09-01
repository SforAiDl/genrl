import numpy as np
import torch
from scipy.stats import invgamma

from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.agents.bandits.contextual.common import NeuralBanditModel, TransitionDB
from genrl.utils.data_bandits.base import DataBasedBandit


class NeuralLinearPosteriorAgent(DCBAgent):
    """Deep contextual bandit agent using bayesian regression on for posterior inference

    A neural network is used to transform context vector to a latent represntation on
    which bayesian regression is performed.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Defaults to [50, 50].
        init_lr (float, optional): Initial learning rate. Defaults to 0.1.
        lr_decay (float, optional): Decay rate for learning rate. Defaults to 0.5.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to True.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
            Defaults to 0.5.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Defaults to False.
        nn_update_ratio (int, optional): . Defaults to 2.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        a0 (float, optional): Inverse gamma prior for noise. Defaults to 3.0.
        b0 (float, optional): Inverse gamma prior for noise. Defaults to 3.0.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(NeuralLinearPosteriorAgent, self).__init__(
            bandit, kwargs.get("device", "cpu")
        )
        self.init_pulls = kwargs.get("init_pulls", 3)
        self.lambda_prior = kwargs.get("lambda_prior", 0.25)
        self.a0 = kwargs.get("a0", 6.0)
        self.b0 = kwargs.get("b0", 6.0)
        hidden_dims = kwargs.get("hidden_dims", [50, 50])
        self.latent_dim = hidden_dims[-1]
        self.nn_update_ratio = kwargs.get("nn_update_ratio", 2)
        self.model = (
            NeuralBanditModel(
                context_dim=self.context_dim,
                hidden_dims=kwargs.get("hidden_dims", [50, 50]),
                n_actions=self.n_actions,
                init_lr=kwargs.get("init_lr", 0.1),
                max_grad_norm=kwargs.get("max_grad_norm", 0.5),
                lr_decay=kwargs.get("lr_decay", 0.5),
                lr_reset=kwargs.get("lr_reset", True),
                dropout_p=kwargs.get("dropout_p", None),
            )
            .to(torch.float)
            .to(self.device)
        )
        self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
        self.mu = torch.zeros(
            size=(self.n_actions, self.latent_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.cov = torch.stack(
            [
                (1.0 / self.lambda_prior)
                * torch.eye(self.latent_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.latent_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.a = self.a0 * torch.ones(
            self.n_actions, device=self.device, dtype=torch.float
        )
        self.b = self.b0 * torch.ones(
            self.n_actions, device=self.device, dtype=torch.float
        )
        self.db = TransitionDB(self.device)
        self.latent_db = TransitionDB()
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by computing a forward pass through network to output
        a representation of the context on which bayesian linear regression is
        performed to select an action.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(
                self.t % self.n_actions, device=self.device, dtype=torch.int
            )
        var = torch.tensor(
            [self.b[i] * invgamma.rvs(self.a[i]) for i in range(self.n_actions)],
            device=self.device,
            dtype=torch.float,
        )
        try:
            beta = (
                torch.tensor(
                    np.stack(
                        [
                            np.random.multivariate_normal(
                                self.mu[i], var[i] * self.cov[i]
                            )
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        except np.linalg.LinAlgError as e:  # noqa F841
            beta = (
                (
                    torch.stack(
                        [
                            torch.distributions.MultivariateNormal(
                                torch.zeros(self.latent_dim + 1),
                                torch.eye(self.latent_dim + 1),
                            ).sample()
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        results = self.model(context)
        latent_context = results["x"]
        values = torch.mv(beta, torch.cat([latent_context.squeeze(0), torch.ones(1)]))
        action = torch.argmax(values).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Updates latent context and predicted rewards seperately.

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db.add(context, action, reward)
        results = self.model(context)
        self.latent_db.add(results["x"].detach(), action, reward)

    def update_params(self, action: int, batch_size: int = 512, train_epochs: int = 20):
        """Update parameters of the agent.

        Trains neural network and updates bayesian regression parameters.

        Args:
            action (int): Action to update the parameters for.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (int, optional): Epochs to train neural network for.
                Defaults to 20
        """
        self.update_count += 1

        if self.update_count % self.nn_update_ratio == 0:
            self.model.train_model(self.db, train_epochs, batch_size)

        z, y = self.latent_db.get_data_for_action(action, batch_size)
        z = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1)
        inv_cov = torch.mm(z.T, z) + self.lambda_prior * torch.eye(self.latent_dim + 1)
        cov = torch.inverse(inv_cov)
        mu = torch.mm(cov, torch.mm(z.T, y))
        a = self.a0 + self.t / 2
        b = self.b0 + (torch.mm(y.T, y) - torch.mm(mu.T, torch.mm(inv_cov, mu))) / 2
        self.mu[action] = mu.squeeze(1)
        self.cov[action] = cov
        self.inv_cov[action] = inv_cov
        self.a[action] = a
        self.b[action] = b
