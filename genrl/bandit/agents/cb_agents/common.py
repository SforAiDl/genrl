import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionDB(object):
    """
    Database for storing (context, action, reward) transitions.

    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        db (dict): Dictionary containing list of transitions.
        db_size (int): Number of transitions stored in database.
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):

        if type(device) is str:
            if "cuda" in device and torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.db = {"contexts": [], "actions": [], "rewards": []}
        self.db_size = 0

    def add(self, context: torch.Tensor, action: int, reward: int):
        """Add (context, action, reward) transition to database

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db["contexts"].append(context)
        self.db["actions"].append(action)
        self.db["rewards"].append(reward)
        self.db_size += 1

    def get_data(
        self, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database

        Args:
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts, actions, rewards tensors.
        """
        if batch_size is None:
            batch_size = self.db_size
        else:
            batch_size = min(batch_size, self.db_size)
        idx = [random.randrange(self.db_size) for _ in range(batch_size)]
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        a = (
            torch.stack([self.db["actions"][i] for i in idx])
            .to(self.device)
            .to(torch.long)
        )
        return x, a, y

    def get_data_for_action(
        self, action: int, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database for a given action.

        Args:
            action (int): The action to sample transitions for.
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts and rewards tensors.
        """
        action_idx = [i for i in range(self.db_size) if self.db["actions"][i] == action]
        if batch_size is None:
            t_batch_size = len(action_idx)
        else:
            t_batch_size = min(batch_size, len(action_idx))
        idx = random.sample(action_idx, t_batch_size)
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        return x, y


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
        use_dropout (int): Indicated whether or not dropout should be used in forward pass.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dims: List[int],
        n_actions: int,
        init_lr: float,
        max_grad_norm: float,
        lr_decay: Optional[float] = None,
        lr_reset: bool = False,
        dropout_p: Optional[float] = None,
    ):
        super(NeuralBanditModel, self).__init__()
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        t_hidden_dims = [context_dim, *hidden_dims, n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(nn.Linear(t_hidden_dims[i], t_hidden_dims[i + 1]))
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=init_lr)
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        if self.lr_decay is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
            )
        self.lr_reset = lr_reset
        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.use_dropout = True
        self.max_grad_norm = max_grad_norm

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


class BayesianLinear(nn.Module):
    """Linear Layer for Bayesian Neural Networks.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool, optional): Whether to use an additive bias. Defaults to True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_features))
            self.b_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets weight and bias parameters of the layer.
        """
        self.w_mu.data.normal_(0, 0.1)
        self.w_sigma.data.normal_(0, 0.1)
        if self.bias:
            self.b_mu.data.normal_(0, 0.1)
            self.b_sigma.data.normal_(0, 0.1)

    def forward(
        self, x: torch.Tensor, kl: bool = True, frozen: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply linear transormation to input.

        The weight and bias is sampled for each forward pass from a normal
        distribution. The KL divergence of the sampled weigth and bias can
        also be computed if specified.

        Args:
            x (torch.Tensor): Input to be transformed
            kl (bool, optional): Whether to compute the KL divergence. Defaults to True.
            frozen (bool, optional): Whether to freeze current parameters. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed input and optionally
                the computed KL divergence value.
        """
        kl_val = None
        b = None
        if frozen:
            w = self.w_mu
            if self.bias:
                b = self.b_mu
        else:
            w_dist = torch.distributions.Normal(self.w_mu, self.w_sigma)
            w = w_dist.rsample()
            if kl:
                kl_val = torch.sum(
                    w_dist.log_prob(w) - torch.distributions.Normal(0, 0.1).log_prob(w)
                )
            if self.bias:
                b_dist = torch.distributions.Normal(self.b_mu, self.b_sigma)
                b = b_dist.rsample()
                if kl:
                    kl_val += torch.sum(
                        b_dist.log_prob(b)
                        - torch.distributions.Normal(0, 0.1).log_prob(b)
                    )
            else:
                b = 0.0
                # b_logprob = None

        return F.linear(x, w, b), kl_val


class BayesianNNBanditModel(nn.Module):
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

    def __init__(
        self,
        context_dim: int,
        hidden_dims: List[int],
        n_actions: int,
        init_lr: float,
        max_grad_norm: float,
        lr_decay: Optional[float] = None,
        lr_reset: bool = False,
        dropout_p: Optional[float] = None,
        noise_std: float = 0.1,
    ):
        super(BayesianNNBanditModel, self).__init__()
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.noise_std = noise_std
        t_hidden_dims = [context_dim, *hidden_dims, n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(BayesianLinear(t_hidden_dims[i], t_hidden_dims[i + 1]))
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=init_lr)
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        if self.lr_decay is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda i: 1 / (1 + self.lr_decay * i)
            )
        self.lr_reset = lr_reset
        self.dropout_p = dropout_p
        if self.dropout_p is not None:
            self.use_dropout = True
        self.max_grad_norm = max_grad_norm

    def forward(
        self, context: torch.Tensor, kl: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Computes forward pass through the network.

        Args:
            context (torch.Tensor): The context vector to perform forward pass on.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple with the output
                of the second to last layer of the network, the final output of the
                network and the value of accumulated kl divergence.
        """
        kl_val = 0.0
        x = context
        for layer in self.layers[:-1]:
            x, kl_v = layer(x)
            x = F.relu(x)
            if self.dropout_p is not None and self.use_dropout is True:
                x = F.dropout(x, p=self.dropout_p)
            if kl:
                kl_val += kl_v
        pred_rewards, kl_v = self.layers[-1](x)
        if kl:
            kl_val += kl_v
        return x, pred_rewards, kl_val

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
                size=(y.shape[0], self.n_actions), dtype=torch.float
            )
            reward_vec[:, a] = y.view(-1)
            _, rewards_pred, kl_val = self.forward(x)
            action_mask = F.one_hot(a, num_classes=self.n_actions)

            log_likelihood = torch.distributions.Normal(
                rewards_pred, self.noise_std
            ).log_prob(reward_vec)

            loss = torch.sum(action_mask * log_likelihood) / batch_size - (
                kl_val / db.db_size
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_decay is not None:
                self.lr_scheduler.step()
