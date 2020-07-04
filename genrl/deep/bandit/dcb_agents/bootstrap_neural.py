from typing import List

import torch
import numpy as np

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class BootstrapNeural(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        n: int = 1,
        add_prob: float = 0.95,
        init_pulls: int = 2,
        hidden_dims: List[int] = [100, 100],
        train_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 512,
        nn_update_interval: int = 20,
    ):
        super(BootstrapNeural, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.n = n
        self.add_prob = add_prob
        self.models = [
            NeuralBanditModel(self.context_dim, hidden_dims, self.n_actions, lr)
            for _ in range(n)
        ]
        self.dbs = [TransitionDB() for _ in range(n)]
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.nn_update_interval = nn_update_interval
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        _, predicted_rewards = self.models[np.random.randint(self.n)](context)
        return torch.argmax(predicted_rewards).to(torch.int)

    def update_params(self, context: torch.Tensor, action: int, reward: int):
        self.update_count += 1

        for db in self.dbs:
            if np.random.random() < self.add_prob or self.update_count <= 1:
                db.add(context, action, reward)

        if self.update_count % self.nn_update_interval == 0:
            for i, model in enumerate(self.models):
                model.train(self.dbs[i], self.train_epochs, self.batch_size)


if __name__ == "__main__":

    from .common import demo_dcb_policy
    from ..data_bandits.covertype_bandit import CovertypeDataBandit
    from ..data_bandits.mushroom_bandit import MushroomDataBandit
    from ..data_bandits.statlog_bandit import StatlogDataBandit
    from ....classical.bandit.contextual_bandits import BernoulliCB

    TIMESTEPS = 1000
    ITERATIONS = 100
    # BANDIT_ARGS = {"download": True}
    BANDIT_ARGS = {"bandits": 10, "arms": 10}

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
        BootstrapNeural,
        BernoulliCB,
        POLICY_ARGS_COLLECTION,
        BANDIT_ARGS,
        TIMESTEPS,
        ITERATIONS,
        verbose=True,
    )
