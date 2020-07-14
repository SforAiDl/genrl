
import torch

from ..data_bandits import DataBasedBandit


class DCBAgent:
    def __init__(self, bandit: DataBasedBandit):
        self._bandit = bandit
        self.context_dim = self._bandit.context_dim
        self.n_actions = self._bandit.n_actions
        self._action_hist = []

    def select_action(self, context: torch.Tensor) -> int:
        raise NotImplementedError

    def update_parameters(
        self,
        context: torch.Tensor,
        action: int,
        reward: int,
        batch_size: int,
        train_epochs: int,
    ):
        raise NotImplementedError
