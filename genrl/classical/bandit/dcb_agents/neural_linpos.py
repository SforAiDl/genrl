from typing import List

import torch
from scipy.stats import invgamma

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class NeuralLinearPosteriorAgent(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 2,
        lambda_prior: float = 0.5,
        a0: float = 0.0,
        b0: float = 0.0,
        hidden_dims: List[int] = [128],
        train_epochs: int = 1,
        lr: float = 1e-3,
        bayesian_update_interval: int = 1,
        nn_update_interval: int = 1,
    ):
        super(NeuralLinearPosteriorAgent, self).__init__(bandit)
        self.context_dim = self._bandit.context_dim
        self.n_actions = self._bandit.n_actions
        self.init_pulls = init_pulls
        self.lambda_prior = lambda_prior
        self.a0 = a0
        self.b0 = b0
        self.latent_dim = hidden_dims[-1]
        self.model = NeuralBanditModel(
            self.context_dim, hidden_dims, self.n_actions, lr
        )
        self.train_epochs = train_epochs
        self.mu = torch.zeros(
            size=(self.n_actions, self.latent_dim + 1), device=device, dtype=dtype
        )
        self.cov = torch.stack(
            [
                (1.0 / self.lambda_prior)
                * torch.eye(self.latent_dim + 1, device=device, dtype=dtype)
                for _ in range(self.n_actions)
            ]
        )
        self.inv_cov = torch.stack(
            [
                self.lambda_prior
                * torch.eye(self.latent_dim + 1, device=device, dtype=dtype)
                for _ in range(self.n_actions)
            ]
        )
        self.a = self.a0 * torch.ones(self.n_actions, device=device, dtype=dtype)
        self.b = self.b0 * torch.ones(self.n_actions, device=device, dtype=dtype)
        self.db = TransitionDB()
        self.latent_db = TransitionDB()
        self.nn_update_interval = nn_update_interval
        self.bayesian_update_interval = bayesian_update_interval
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        var = torch.tensor(
            [self.b[i] * invgamma.rvs(self.a[i]) for i in range(self.n_actions)],
            device=device,
            dtype=dtype,
        )
        try:
            beta = (
                torch.stack(
                    [
                        torch.distributions.MultivariateNormal(
                            self.mu[i], var[i] * self.cov[i]
                        ).sample()
                        for i in range(self.n_actions)
                    ]
                )
                .to(device)
                .to(dtype)
            )
        except RuntimeError as e:  # noqa F841
            # print(f"Eror: {e}")
            beta = (
                torch.stack(
                    [
                        torch.distributions.MultivariateNormal(
                            torch.zeros(self.latent_dim + 1),
                            torch.eye(self.latent_dim + 1),
                        ).sample()
                        for i in range(self.n_actions)
                    ]
                )
                .to(device)
                .to(dtype)
            )
        latent_context, _ = self.model(context)
        values = torch.mv(beta, torch.cat([latent_context.squeeze(0), torch.ones(1)]))
        return torch.argmax(values).to(torch.int)

    def update_params(self, context: torch.Tensor, action: int, reward: int):
        self.update_count += 1
        self.reward_hist.append(reward)
        self.db.add(context, action, reward)
        latent_context, _ = self.model(context)
        self.latent_db.add(latent_context, action, reward)

        if self.update_count % self.nn_update_interval == 0:
            x, y = self.db.get_data(action)
            reward_vec = torch.zeros(
                size=(y.shape[0], self.n_actions), device=device, dtype=dtype
            )
            reward_vec[:, action] = y.view(-1)
            self.model.train(x, reward_vec, self.train_epochs)

        if self.update_count % self.bayesian_update_interval == 0:
            z, y = self.latent_db.get_data(action)
            z = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1)
            inv_cov = torch.mm(z.T, z) + self.lambda_prior * torch.eye(
                self.latent_dim + 1
            )
            cov = torch.inverse(inv_cov)
            mu = torch.mm(cov, torch.mm(z.T, y))
            a = self.a0 + self.t / 2
            b = self.b0 + (torch.mm(y.T, y) - torch.mm(mu.T, torch.mm(inv_cov, mu))) / 2
            self.mu[action] = mu.squeeze(1)
            self.cov[action] = cov
            self.inv_cov[action] = inv_cov
            self.a[action] = a
            self.b[action] = b
