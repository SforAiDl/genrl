from abc import ABC, abstractmethod
from typing import Dict

import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F

from genrl.agents.bandits.contextual.common.transition import TransitionDB


class Model(nn.Module, ABC):
    """Bayesian Neural Network used in Deep Contextual Bandit Models.

    Args:
        context_dim (int): Length of context vector.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
        n_actions (int): Number of actions that can be selected. Taken as length
            of output vector for network to predict.
        init_lr (float, optional): Initial learning rate.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
        lr_decay (float, optional): Decay rate for learning rate.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to False.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        noise_std (float): Standard deviation of noise used in the network. Defaults to 0.1

    Attributes:
        use_dropout (int): Indicated whether or not dropout should be used in forward pass.
    """

    def __init__(self, layer, **kwargs):
        super(Model, self).__init__()
        self.context_dim = kwargs.get("context_dim")
        self.hidden_dims = kwargs.get("hidden_dims")
        self.n_actions = kwargs.get("n_actions")
        t_hidden_dims = [self.context_dim, *self.hidden_dims, self.n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(layer(t_hidden_dims[i], t_hidden_dims[i + 1]))
        self.init_lr = kwargs.get("init_lr", 3e-4)
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.init_lr)
        self.lr_decay = kwargs.get("lr_decay", None)
        self.lr_scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
            )
            if self.lr_decay is not None
            else None
        )
        self.lr_reset = kwargs.get("lr_reset", False)
        self.dropout_p = kwargs.get("dropout_p", None)
        self.use_dropout = True if self.dropout_p is not None else False
        self.max_grad_norm = kwargs.get("max_grad_norm")

    @abstractmethod
    def forward(self, context: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Computes forward pass through the network.

        Args:
            context (torch.Tensor): The context vector to perform forward pass on.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of outputs
        """

    def train_model(self, db: TransitionDB, epochs: int, batch_size: int):
        """Trains the network on a given database for given epochs and batch_size.

        Args:
            db (TransitionDB): The database of transitions to train on.
            epochs (int): Number of gradient steps to take.
            batch_size (int): The size of each batch to perform gradient descent on.
        """
        self.use_dropout = True if self.dropout_p is not None else False

        if self.lr_decay is not None and self.lr_reset is True:
            self._reset_lr(self.init_lr)

        for _ in range(epochs):
            x, a, y = db.get_data(batch_size)
            action_mask = F.one_hot(a, num_classes=self.n_actions)
            reward_vec = y.view(-1).repeat(self.n_actions, 1).T * action_mask
            loss = self._compute_loss(db, x, action_mask, reward_vec, batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_decay is not None:
                self.lr_scheduler.step()

    @abstractmethod
    def _compute_loss(
        self,
        db: TransitionDB,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        reward_vec: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Computes loss for the model

        Args:
            db (TransitionDB): The database of transitions to train on.
            x (torch.Tensor): Context.
            action_mask (torch.Tensor): Mask of actions taken.
            reward_vec (torch.Tensor): Reward vector recieved.
            batch_size (int): The size of each batch to perform gradient descent on.

        Returns:
            torch.Tensor: The computed loss.
        """

    def _reset_lr(self, lr: float) -> None:
        """Resets learning rate of optimizer.

        Args:
            lr (float): New value of learning rate
        """
        for o in self.optimizer.param_groups:
            o["lr"] = lr
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
            )
