from typing import List, Optional

import numpy as np
import torch
from torch import dropout

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class NeuralNoiseSamplingAgent(DCBAgent):
    """Deep contextual bandit agent with noise sampling for neural network parameters.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Defaults to [50, 50].
        init_lr (float, optional): Initial learning rate. Defaults to 0.1.
        lr_decay (float, optional): Decay rate for learning rate. Defaults to 0.5.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to True.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
            Defaults to 0.5.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Defaults to False.
        noise_std_dev (float, optional): Standard deviation of sampled noise.
            Defaults to 0.05.
        eps (float, optional): Small constant for bounding KL divergece of noise.
            Defaults to 0.1.
        noise_update_batch_size (int, optional): Batch size for updating noise parameters.
            Defaults to 256.
    """

    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 3,
        hidden_dims: List[int] = [50, 50],
        init_lr: float = 0.1,
        lr_decay: float = 0.5,
        lr_reset: bool = True,
        max_grad_norm: float = 0.5,
        dropout_p: Optional[float] = None,
        eval_with_dropout: bool = False,
        noise_std_dev: float = 0.05,
        eps: float = 0.1,
        noise_update_batch_size: int = 256,
    ):
        super(NeuralNoiseSamplingAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = NeuralBanditModel(
            self.context_dim,
            hidden_dims,
            self.n_actions,
            init_lr,
            max_grad_norm,
            lr_decay,
            lr_reset,
            dropout_p,
        )
        self.eval_with_dropout = eval_with_dropout
        self.noise_std_dev = noise_std_dev
        self.eps = eps
        self.db = TransitionDB()
        self.noise_update_batch_size = noise_update_batch_size
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by adding noise to neural network paramters and
        the computing forward with the context vector as input.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)

        _, predicted_rewards = self._noisy_pred(context)
        action = torch.argmax(predicted_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db.add(context, action, reward)

    def update_params(
        self,
        batch_size: int = 512,
        train_epochs: int = 100,
        action: Optional[int] = None,
    ):
        """Update parameters of the agent.

        Trains neural network and updates noise parameters.

        Args:
            batch_size (int): Size of batch to update parameters with.
            train_epochs (int): Epochs to train neural network for.
            action (Optional[int], optional): Action to update the parameters for.
                Defaults to None.
        """
        self.update_count += 1
        self.model.train_model(self.db, train_epochs, batch_size)
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
