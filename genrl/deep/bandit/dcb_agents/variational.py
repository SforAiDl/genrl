from typing import List

import torch
from scipy.stats import invgamma

from ..data_bandits import DataBasedBandit
from .common import BayesianNNBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class VariationalAgent(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 2,
        hidden_dims: List[int] = [128],
        lr: float = 1e-3,
        noise_sigma: float = 0.1,
    ):
        super(VariationalAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = BayesianNNBanditModel(
            self.context_dim, hidden_dims, self.n_actions, lr, noise_sigma
        )
        self.db = TransitionDB()
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        _, predicted_rewards, _ = self.model(context)
        action = torch.argmax(predicted_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        self.db.add(context, action, reward)

    def update_params(
        self,
        context: torch.Tensor,
        action: int,
        reward: int,
        train_epochs: int = 20,
        batch_size: int = 512,
    ):
        self.update_count += 1
        self.model.train(self.db, train_epochs, batch_size)
