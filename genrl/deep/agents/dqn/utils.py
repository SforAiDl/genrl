import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDQN


def ddqn_q_target(
    agent: BaseDQN,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    next_q_values = agent.model(next_states)
    next_best_actions = next_q_values.max(1)[1].unsqueeze(1)

    rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

    next_q_target_values = agent.target_model(next_states)
    max_next_q_target_values = next_q_target_values.gather(1, next_best_actions)
    target_q_values = rewards + agent.gamma * torch.mul(
        max_next_q_target_values, (1 - dones)
    )
    return target_q_values


def noisy_mlp(fc_layers: List[int], noisy_layers: List[int]):
    model = []
    for layer in range(len(fc_layers) - 1):
        model += [nn.Linear(fc_layers[layer], fc_layers[layer + 1]), nn.ReLU()]
    for layer in range(len(noisy_layers) - 1):
        model += [NoisyLinear(noisy_layers[layer], noisy_layers[layer + 1])]
        if layer < len(noisy_layers) - 2:
            model += [nn.ReLU()]
    return nn.Sequential(*model)


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(state, weight, bias)

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        inp = torch.randn(size)
        inp = inp.sign().mul(inp.abs().sqrt())
        return inp
