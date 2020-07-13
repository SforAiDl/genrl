from typing import List

import numpy as np
import torch

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class BootstrapNeuralAgent(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        n: int = 10,
        add_prob: float = 0.95,
        init_pulls: int = 2,
        hidden_dims: List[int] = [100, 100],
        lr: float = 1e-2,
    ):
        super(BootstrapNeuralAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.n = n
        self.add_prob = add_prob
        self.models = [
            NeuralBanditModel(self.context_dim, hidden_dims, self.n_actions, lr)
            for _ in range(n)
        ]
        self.dbs = [TransitionDB() for _ in range(n)]
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        _, predicted_rewards = self.models[np.random.randint(self.n)](context)
        action = torch.argmax(predicted_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        for db in self.dbs:
            if np.random.random() < self.add_prob or self.update_count <= 1:
                db.add(context, action, reward)

    def update_params(
        self,
        context: torch.Tensor,
        action: int,
        reward: int,
        train_epochs: int = 20,
        batch_size: int = 512,
    ):
        self.update_count += 1
        for i, model in enumerate(self.models):
            model.train(self.dbs[i], train_epochs, batch_size)
