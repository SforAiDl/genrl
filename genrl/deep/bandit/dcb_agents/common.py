import random
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ..data_bandits import DataBasedBandit
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class TransitionDB(object):
    def __init__(self):
        self.db = {"contexts": [], "actions": [], "rewards": []}
        self.db_size = 0

    def add(self, context: torch.Tensor, action: int, reward: int):
        self.db["contexts"].append(context)
        self.db["actions"].append(action)
        self.db["rewards"].append(reward)
        self.db_size += 1

    def get_data(
        self, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.db_size
        else:
            batch_size = min(batch_size, self.db_size)
        idx = [random.randrange(self.db_size) for _ in range(batch_size)]
        x = torch.stack([self.db["contexts"][i] for i in idx]).to(device).to(dtype)
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(device)
            .to(dtype)
            .unsqueeze(1)
        )
        a = torch.stack([self.db["actions"][i] for i in idx]).to(device).to(torch.long)
        return x, a, y

    def get_data_for_action(
        self, action: int, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action_idx = [i for i in range(self.db_size) if self.db["actions"][i] == action]
        if batch_size is None:
            batch_size = len(action_idx)
        else:
            batch_size = min(batch_size, self.db_size)
        idx = random.sample(action_idx, batch_size)
        x = torch.stack([self.db["contexts"][i] for i in idx]).to(device).to(dtype)
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(device)
            .to(dtype)
            .unsqueeze(1)
        )
        return x, y


class NeuralBanditModel(nn.Module):
    def __init__(
        self, context_dim: int, hidden_dims: List[int], n_actions: int, lr: float,
    ):
        super(NeuralBanditModel, self).__init__()
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        hidden_dims.insert(0, context_dim)
        hidden_dims.append(n_actions)
        self.layers = nn.ModuleList([])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = context
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        pred_rewards = self.layers[-1](x)
        return x, pred_rewards

    def train(self, db: TransitionDB, epochs: int, batch_size: int):
        for e in range(epochs):
            x, a, y = db.get_data(batch_size)
            reward_vec = torch.zeros(
                size=(y.shape[0], self.n_actions), device=device, dtype=dtype
            )
            reward_vec[:, a] = y.view(-1)
            _, rewards_pred = self.forward(x)
            action_mask = F.one_hot(a, num_classes=self.n_actions)
            loss = (
                torch.sum(action_mask * (reward_vec - rewards_pred) ** 2) / batch_size
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class BayesianLinear(nn.Module):
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
        self.w_mu.data.normal_(0, 0.1)
        self.w_sigma.data.normal_(0, 0.1)
        if self.bias:
            self.b_mu.data.normal_(0, 0.1)
            self.b_sigma.data.normal_(0, 0.1)

    def forward(
        self, x: torch.Tensor, kl: bool = True, frozen: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kl_val = None
        if frozen:
            w = self.w_mu
            if bias:
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
                b_logprob = None

        return F.linear(x, w, b), kl_val


class BayesianNNBanditModel(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dims: List[int],
        n_actions: int,
        lr: float,
        noise_sigma: float = 0.1,
    ):
        super(BayesianNNBanditModel, self).__init__()
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.noise_sigma = noise_sigma
        hidden_dims.insert(0, context_dim)
        hidden_dims.append(n_actions)
        self.layers = nn.ModuleList([])
        for i in range(len(hidden_dims) - 1):
            self.layers.append(BayesianLinear(hidden_dims[i], hidden_dims[i + 1]))
        self.optimizer = torch.optim.Adam(self.layers.parameters(), lr=lr)

    def forward(
        self, context: torch.Tensor, kl: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        kl_val = 0.0
        x = context
        for layer in self.layers[:-1]:
            x, kl_v = layer(x)
            x = F.relu(x)
            if kl:
                kl_val += kl_v
        pred_rewards, kl_v = self.layers[-1](x)
        if kl:
            kl_val += kl_v
        return x, pred_rewards, kl_val

    def train(self, db: TransitionDB, epochs: int, batch_size: int):
        for e in range(epochs):
            x, a, y = db.get_data(batch_size)
            reward_vec = torch.zeros(
                size=(y.shape[0], self.n_actions), device=device, dtype=dtype
            )
            reward_vec[:, a] = y.view(-1)
            _, rewards_pred, kl_val = self.forward(x)
            action_mask = F.one_hot(a, num_classes=self.n_actions)

            log_likelihood = torch.distributions.Normal(
                rewards_pred, self.noise_sigma
            ).log_prob(reward_vec)

            loss = torch.sum(action_mask * log_likelihood) / batch_size - (
                kl_val / db.db_size
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def demo_dcb_policy(
    policy_type: DCBAgent,
    bandit_type: DataBasedBandit,
    policy_args_collection: Dict[str, Any],
    bandit_args: Dict[str, Any],
    timesteps: int,
    iterations: int,
    verbose: bool = False,
):
    """ Plots rewards and regrets of a given policy on given bandit """

    print(f"\nRunning {policy_type.__name__} on {bandit_type.__name__}")
    _, axs = plt.subplots(1, 3, figsize=(15, 4))
    for policy_args in policy_args_collection:
        print(f"Running with policy parameters: = {policy_args}")
        average_reward = torch.zeros(timesteps)
        average_regret = torch.zeros(timesteps)
        for i in range(iterations):
            try:
                if verbose:
                    print(f"Iteration {i + 1}")
                bandit = bandit_type(**bandit_args)
                policy = policy_type(bandit, **policy_args)
                policy.learn(timesteps)
                average_reward += torch.tensor(bandit.reward_hist) / float(iterations)
                average_regret += torch.tensor(bandit.regret_hist) / float(iterations)
            except KeyboardInterrupt:
                print("KeyboardInterrupt recieved!")
                break
        axs[0].plot(average_reward)
        axs[1].plot(average_regret)
        axs[2].plot(torch.cumsum(average_regret, dim=0))
    axs[0].set_title("Rewards")
    axs[1].set_title("Regrets")
    axs[2].set_title("Cumulative Regrets")
    plt.savefig(f"./logs/{policy_type.__name__}-on-{bandit_type.__name__}.png")
    plt.cla()
    print(f"Simple Regret: {torch.mean(average_regret[-500:]):.4f}")
    print(f"Simple Reward: {torch.mean(average_reward[-500:]):.4f}")
