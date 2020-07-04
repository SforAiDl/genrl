from typing import List

import torch

from ..data_bandits import DataBasedBandit


class DCBAgent:
    def __init__(self, bandit: DataBasedBandit):
        self._bandit = bandit
        self.context_dim = self._bandit.context_dim
        self.n_actions = self._bandit.n_actions
        self._action_hist = []

    @property
    def action_hist(self) -> List[float]:
        """
        Get the history of action taken at each step
        :returns: List of actions
        :rtype: list
        """
        return self._action_hist

    def learn(self, n_timesteps: int):
        context = self._bandit.reset()
        for _ in range(n_timesteps):
            action = self.select_action(context)
            context, reward = self._bandit.step(action)
            self.action_hist.append(action)
            self.update_params(context, action, reward)

    def select_action(self, context: torch.Tensor) -> int:
        raise NotImplementedError
