from typing import Optional

import numpy as np
import torch
from scipy.stats import invgamma

from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.agents.bandits.contextual.common import TransitionDB
from genrl.utils.data_bandits.base import DataBasedBandit


class LinearPosteriorAgent(DCBAgent):
    """Deep contextual bandit agent using bayesian regression for posterior inference.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        lambda_prior (float, optional): Guassian prior for linear model. Defaults to 0.25.
        a0 (float, optional): Inverse gamma prior for noise. Defaults to 6.0.
        b0 (float, optional): Inverse gamma prior for noise. Defaults to 6.0.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(LinearPosteriorAgent, self).__init__(bandit, kwargs.get("device", "cpu"))

        self.init_pulls = kwargs.get("init_pulls", 3)
        self.lambda_prior = kwargs.get("lambda_prior", 0.25)
        self.a0 = kwargs.get("a0", 6.0)
        self.b0 = kwargs.get("b0", 6.0)
        self.mu = torch.zeros(
            size=(self.n_actions, self.context_dim + 1),
            device=self.device,
            dtype=torch.float,
        )
        self.cov = torch.stack(
            [
                (1.0 / self.lambda_prior)
                * torch.eye(self.context_dim + 1, device=self.device, dtype=torch.float)
                for _ in range(self.n_actions)
            ]
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.context_dim + 1, device=self.device, dtype=torch.float)
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
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selecting action with highest predicted reward computed through
        betas sampled from posterior.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
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
                                torch.zeros(self.context_dim + 1),
                                torch.eye(self.context_dim + 1),
                            ).sample()
                            for i in range(self.n_actions)
                        ]
                    )
                )
                .to(self.device)
                .to(torch.float)
            )
        values = torch.mv(beta, torch.cat([context.view(-1), torch.ones(1)]))
        action = torch.argmax(values).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db.add(context, action, reward)

    def update_params(
        self, action: int, batch_size: int = 512, train_epochs: Optional[int] = None
    ):
        """Update parameters of the agent.

        Updated the posterior over beta though bayesian regression.

        Args:
            action (int): Action to update the parameters for.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (Optional[int], optional): Epochs to train neural network for.
                Not applicable in this agent. Defaults to None
        """
        self.update_count += 1

        x, y = self.db.get_data_for_action(action, batch_size)
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        inv_cov = torch.mm(x.T, x) + self.lambda_prior * torch.eye(self.context_dim + 1)
        cov = torch.pinverse(inv_cov)
        mu = torch.mm(cov, torch.mm(x.T, y))
        a = self.a0 + self.t / 2
        b = self.b0 + (torch.mm(y.T, y) - torch.mm(mu.T, torch.mm(inv_cov, mu))) / 2
        self.mu[action] = mu.squeeze(1)
        self.cov[action] = cov
        self.inv_cov[action] = inv_cov
        self.a[action] = a
        self.b[action] = b
