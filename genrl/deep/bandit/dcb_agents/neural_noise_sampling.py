from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class NeuralNoiseSamplingAgent(DCBAgent):
    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 2,
        hidden_dims: List[int] = [100, 100],
        train_epochs: int = 20,
        lr: float = 1e-3,
        noise_std_dev: float = 0.05,
        eps: float = 0.1,
        batch_size: int = 128,
        noise_update_batch_size: int = 64,
        update_interval: int = 20,
    ):
        super(NeuralNoiseSamplingAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = NeuralBanditModel(
            self.context_dim, hidden_dims, self.n_actions, lr
        )
        self.noise_std_dev = noise_std_dev
        self.eps = eps
        self.train_epochs = train_epochs
        self.db = TransitionDB()
        self.batch_size = batch_size
        self.noise_update_batch_size = noise_update_batch_size
        self.update_interval = update_interval
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)

        _, predicted_rewards = self._noisy_pred(context)

        return torch.argmax(predicted_rewards).to(torch.int)

    def update_params(self, context: torch.Tensor, action: int, reward: int):
        self.update_count += 1
        self.db.add(context, action, reward)

        if self.update_count % self.update_interval == 0:
            self.model.train(self.db, self.train_epochs, self.batch_size)
            self._update_noise()

    def _noisy_pred(self, context: torch.Tensor) -> torch.Tensor:
        noise = []
        with torch.no_grad():
            for p in self.model.parameters():
                noise.append(torch.normal(0, self.noise_std_dev, size=p.shape))
                p += noise[-1]

        x, predicted_rewards = self.model(context)

        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                p -= noise[i]

        return x, predicted_rewards

    def _update_noise(self):
        x, _, _ = self.db.get_data(self.noise_update_batch_size)
        with torch.no_grad():
            y_pred, _ = self.model(x)
            y_pred_noisy, _ = self._noisy_pred(x)

        p = torch.distributions.Categorical(logits=y_pred)
        q = torch.distributions.Categorical(logits=y_pred_noisy)
        kl = torch.distributions.kl.kl_divergence(p, q).mean()

        delta = -np.log1p(-self.eps + self.eps / self.n_actions)
        if kl < delta:
            self.noise_std_dev *= 1.01
        else:
            self.noise_std_dev /= 1.01

        self.eps *= 0.99


if __name__ == "__main__":

    from .common import demo_dcb_policy
    from ..data_bandits.covertype_bandit import CovertypeDataBandit
    from ..data_bandits.mushroom_bandit import MushroomDataBandit
    from ..data_bandits.statlog_bandit import StatlogDataBandit
    from ....classical.bandit.contextual_bandits import BernoulliCB

    TIMESTEPS = 1000
    ITERATIONS = 30
    # BANDIT_ARGS = {"download": True}
    BANDIT_ARGS = {"bandits": 10, "arms": 10}

    POLICY_ARGS_COLLECTION = [
        {
            "init_pulls": 2,
            "hidden_dims": [100, 100],
            "train_epochs": 20,
            "lr": 1e-3,
            "batch_size": 64,
            "update_interval": 20,
        }
    ]

    demo_dcb_policy(
        NeuralNoiseSamplingAgent,
        BernoulliCB,
        POLICY_ARGS_COLLECTION,
        BANDIT_ARGS,
        TIMESTEPS,
        ITERATIONS,
        verbose=True,
    )
