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
        lr: float = 1e-3,
        noise_std_dev: float = 0.05,
        eps: float = 0.1,
        noise_update_batch_size: int = 64,
    ):
        super(NeuralNoiseSamplingAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = NeuralBanditModel(
            self.context_dim, hidden_dims, self.n_actions, lr
        )
        self.noise_std_dev = noise_std_dev
        self.eps = eps
        self.db = TransitionDB()
        self.noise_update_batch_size = noise_update_batch_size
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)

        _, predicted_rewards = self._noisy_pred(context)
        action = torch.argmax(predicted_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        self.db.add(context, action, reward)

    def update_params(
        self,
        context: torch.Tensor,
        action: int,
        reward: int,
        batch_size: int = 512,
        train_epochs: int = 20,
    ):
        self.update_count += 1
        self.model.train(self.db, train_epochs, batch_size)
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
