from typing import List

import numpy as np
import torch

from ..data_bandits import DataBasedBandit
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class FixedAgent(DCBAgent):
    def __init__(self, bandit: DataBasedBandit, p: List[float] = None):
        super(FixedAgent, self).__init__(bandit)
        if p is None:
            p = [1 / self.n_actions for _ in range(self.n_actions)]
        elif len(p) != self.n_actions:
            raise ValueError(f"p should be of length {self.n_actions}")
        self.p = p
        self.t = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        return np.random.choice(range(self.n_actions), p=self.p)

    def update_db(self, *args, **kwargs):
        pass

    def update_params(self, *args, **kwargs):
        pass
