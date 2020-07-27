from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrl.bandit.agents.cb_agents.common.transition import TransitionDB


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

    def __init__(self, **kwargs):
        super(BayesianNNBanditModel, self).__init__()
        self.context_dim = kwargs.get("context_dim")
        self.hidden_dims = kwargs.get("hidden_dims")
        self.n_actions = kwargs.get("n_actions")
        self.noise_std = kwargs.get("noise_std", 0.1)
        t_hidden_dims = [self.context_dim, *self.hidden_dims, self.n_actions]
        self.layers = nn.ModuleList([])
        for i in range(len(t_hidden_dims) - 1):
            self.layers.append(BayesianLinear(t_hidden_dims[i], t_hidden_dims[i + 1]))
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
