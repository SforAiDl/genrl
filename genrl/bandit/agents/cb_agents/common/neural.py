from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrl.bandit.agents.cb_agents.common.transition import TransitionDB


class NeuralBanditModel(nn.Module):
    """Neural Network used in Deep Contextual Bandit Models.

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

    Attributes:
        use_dropout (bool): Indicated whether or not dropout should be used in forward pass.
    """

    def __init__(self, **kwargs):
        super(NeuralBanditModel, self).__init__()
        self.context_dim = kwargs.get("context_dim")
        self.hidden_dims = kwargs.get("hidden_dims")
        self.n_actions = kwargs.get("n_actions")
        t_hidden_dims = [self.context_dim, *self.hidden_dims, self.n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(nn.Linear(t_hidden_dims[i], t_hidden_dims[i + 1]))
        self.init_lr = kwargs.get("init_lr")
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=self.init_lr)
        self.lr_decay = kwargs.get("lr_decay", None)
        if self.lr_decay is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
            )
        self.lr_reset = kwargs.get("lr_reset", False)
        self.dropout_p = kwargs.get("dropout_p", None)
        if self.dropout_p is not None:
            self.use_dropout = True
        self.max_grad_norm = kwargs.get("max_grad_norm")

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass through the network.

        Args:
            context (torch.Tensor): The context vector to perform forward pass on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple with the output of the second
                to last layer of the network and the final output of the network.
        """
        x = context
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout_p is not None and self.use_dropout is True:
                x = F.dropout(x, self.dropout_p)

        pred_rewards = self.layers[-1](x)
        return x, pred_rewards

    def train_model(self, db: TransitionDB, epochs: int, batch_size: int):
        """Trains the network on a given database for given epochs and batch_size.

        Args:
            db (TransitionDB): The database of transitions to train on.
            epochs (int): Number of gradient steps to take.
            batch_size (int): The size of each batch to perform gradient descent on.
        """
        if self.dropout_p is not None:
            self.use_dropout = True

        if self.lr_decay is not None and self.lr_reset is True:
            for o in self.optimizer.param_groups:
                o["lr"] = self.init_lr
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
                )
        for _ in range(epochs):
            x, a, y = db.get_data(batch_size)
            reward_vec = torch.zeros(
                size=(y.shape[0], self.n_actions), dtype=torch.dtype
            )
            reward_vec[:, a] = y.view(-1)
            _, rewards_pred = self.forward(x)
            action_mask = F.one_hot(a, num_classes=self.n_actions)
            loss = (
                torch.sum(action_mask * (reward_vec - rewards_pred) ** 2) / batch_size
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_decay is not None:
                self.lr_scheduler.step()
