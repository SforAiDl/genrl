from typing import List

import torch

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class NeuralGreedyAgent(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 2,
        hidden_dims: List[int] = [100, 100],
        lr: float = 1e-3,
    ):
        super(NeuralGreedyAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = NeuralBanditModel(
            self.context_dim, hidden_dims, self.n_actions, lr
        )
        self.db = TransitionDB()
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        _, predicted_rewards = self.model(context)
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
