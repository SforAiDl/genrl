from typing import List

import numpy as np
import torch

from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.utils.data_bandits.base import DataBasedBandit


class FixedAgent(DCBAgent):
    def __init__(
        self, bandit: DataBasedBandit, p: List[float] = None, device: str = "cpu"
    ):
        """A fixed policy agent for deep contextual bandits.

        Args:
            bandit (DataBasedBandit): Bandit to solve.
            p (List[float], optional): List of probabilities for each action.
                Defaults to None which implies action is sampled uniformly.
            device (str): Device to use for tensor operations.
                "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

        Raises:
            ValueError: Raised if length of given probabilities is not
                equal to the number of actions available in given bandit.
        """
        super(FixedAgent, self).__init__(bandit, device)
        if p is None:
            p = [1 / self.n_actions for _ in range(self.n_actions)]
        elif len(p) != self.n_actions:
            raise ValueError(f"p should be of length {self.n_actions}")
        self.p = p
        self.t = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on fixed probabilities.

        Args:
            context (torch.Tensor): The context vector to select action for.
                In this agent, context vector is not considered.

        Returns:
            int: The action to take.
        """
        self.t += 1
        return np.random.choice(range(self.n_actions), p=self.p)

    def update_db(self, *args, **kwargs):
        pass

    def update_params(self, *args, **kwargs):
        pass
