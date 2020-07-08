from typing import List

import torch
import numpy as np

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
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

    def update_params(self, context: torch.Tensor, action: int, reward: int):
        pass


if __name__ == "__main__":

    from ....classical.bandit.contextual_bandits import BernoulliCB
    from ..data_bandits.census_bandit import CensusDataBandit
    from ..data_bandits.covertype_bandit import CovertypeDataBandit
    from ..data_bandits.mushroom_bandit import MushroomDataBandit
    from ..data_bandits.statlog_bandit import StatlogDataBandit
    from .common import demo_dcb_policy

    TIMESTEPS = 1000
    ITERATIONS = 10
    BANDIT_ARGS = {"download": True}
    # BANDIT_ARGS = {"bandits": 10, "arms": 10}

    POLICY_ARGS_COLLECTION = [
        {
            "init_pulls": 2,
            "hidden_dims": [100, 100],
            "train_epochs": 20,
            "lr": 1e-3,
            "batch_size": 64,
            "nn_update_interval": 20,
        }
    ]

    demo_dcb_policy(
        FixedAgent,
        CensusDataBandit,
        POLICY_ARGS_COLLECTION,
        BANDIT_ARGS,
        TIMESTEPS,
        ITERATIONS,
        verbose=True,
    )
