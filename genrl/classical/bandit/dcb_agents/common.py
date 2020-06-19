from typing import List, Tuple, Union

import torch
import torch.nn as nn

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
        self, action: int, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.db_size
        idx = [i for i in range(self.db_size) if self.db["actions"][i] == action]
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
        self, context_dim: int, hidden_dims: List[int], n_actions: int, lr: float
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
            x = nn.functional.relu(layer(x))
        pred_rewards = self.layers[-1](x)
        return x, pred_rewards

    def train(self, context: torch.Tensor, rewards: torch.Tensor, epochs: int):
        for e in range(epochs):
            _, rewards_pred = self.forward(context)
            loss = nn.MSELoss()(rewards, rewards_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
