from typing import Optional

import numpy as np
import torch

from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.agents.bandits.contextual.common import NeuralBanditModel, TransitionDB
from genrl.utils.data_bandits.base import DataBasedBandit


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
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(NeuralNoiseSamplingAgent, self).__init__(
            bandit, kwargs.get("device", "cpu")
        )
        self.init_pulls = kwargs.get("init_pulls", 3)
        self.model = (
            NeuralBanditModel(
                context_dim=self.context_dim,
                hidden_dims=kwargs.get("hidden_dims", [50, 50]),
                n_actions=self.n_actions,
                init_lr=kwargs.get("init_lr", 0.1),
                max_grad_norm=kwargs.get("max_grad_norm", 0.5),
                lr_decay=kwargs.get("lr_decay", 0.5),
                lr_reset=kwargs.get("lr_reset", True),
                dropout_p=kwargs.get("dropout_p", None),
            )
            .to(torch.float)
            .to(self.device)
        )
        self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
        self.noise_std_dev = kwargs.get("noise_std_dev", 0.05)
        self.eps = kwargs.get("eps", 0.1)
        self.db = TransitionDB(self.device)
        self.noise_update_batch_size = kwargs.get("noise_update_batch_size", 256)
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
            return torch.tensor(
                self.t % self.n_actions, device=self.device, dtype=torch.int
            )

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
        action: Optional[int] = None,
        batch_size: int = 512,
        train_epochs: int = 20,
    ):
        """Update parameters of the agent.

        Trains each neural network in the ensemble.

        Args:
            action (Optional[int], optional): Action to update the parameters for.
                Not applicable in this agent. Defaults to None.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (int, optional): Epochs to train neural network for.
                Defaults to 20
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

        results = self.model(context)

        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                p -= noise[i]

        return results["x"], results["pred_rewards"]

    def _update_noise(self):
        x, _, _ = self.db.get_data(self.noise_update_batch_size)
        with torch.no_grad():
            results = self.model(x)
            y_pred_noisy, _ = self._noisy_pred(x)

        p = torch.distributions.Categorical(logits=results["x"])
        q = torch.distributions.Categorical(logits=y_pred_noisy)
        kl = torch.distributions.kl.kl_divergence(p, q).mean()

        delta = -np.log1p(-self.eps + self.eps / self.n_actions)
        if kl < delta:
            self.noise_std_dev *= 1.01
        else:
            self.noise_std_dev /= 1.01

        self.eps *= 0.99
