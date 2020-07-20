from typing import Tuple

import torch
from torch import nn as nn

from ...common import cnn, mlp
from .utils import NoisyLinear, noisy_mlp


class MlpDuelingValue(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple = (128, 128)):
        super(MlpDuelingValue, self).__init__()

        self.feature = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU())

        self.advantage = mlp(list(hidden) + [action_dim])
        self.value = mlp(list(hidden) + [1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean()


class CnnDuelingValue(nn.Module):
    def __init__(self, framestack: int, action_dim: int, fc_layers: Tuple = (512,)):
        super(CnnDuelingValue, self).__init__()

        self.conv, output_size = cnn((framestack, 16, 32))

        self.advantage = mlp([output_size] + list(fc_layers) + [action_dim])
        self.value = mlp([output_size] + list(fc_layers) + [1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.conv(inp)
        inp = inp.view(inp.size(0), -1)
        advantage = self.advantage(inp)
        value = self.value(inp)
        return value + advantage - advantage.mean()


class MlpNoisyValue(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        fc_layers: Tuple = (128,),
        noisy_layers: Tuple = (128, 128),
    ):
        super(MlpNoisyValue, self).__init__()

        self.model = noisy_mlp(
            [state_dim] + list(fc_layers), list(noisy_layers) + [action_dim]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CnnNoisyValue(nn.Module):
    def __init__(
        self,
        framestack: int,
        action_dim: int,
        fc_layers: Tuple = (128,),
        noisy_layers: Tuple = (128, 128),
    ):
        super(CnnNoisyValue, self).__init__()

        self.conv, output_size = cnn((framestack, 16, 32))

        self.model = noisy_mlp(
            [output_size] + list(fc_layers), list(noisy_layers) + [action_dim]
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.conv(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.model(inp)
        return inp

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class MlpCategoricalValue(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        fc_layers: Tuple = (128,),
        noisy_layers: Tuple = (128, 512),
        num_atoms: int = 51,
    ):
        super(MlpCategoricalValue, self).__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.model = noisy_mlp(
            [state_dim] + list(fc_layers),
            list(noisy_layers) + [self.action_dim * self.num_atoms],
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.model(state)
        dist = nn.functional.softmax(features.view(-1, self.num_atoms), dim=0).view(
            -1, self.action_dim, self.num_atoms
        )
        return dist

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CnnCategoricalValue(nn.Module):
    def __init__(
        self,
        framestack: int,
        action_dim: int,
        fc_layers: Tuple = (128, 128),
        noisy_layers: Tuple = (128, 512),
        num_atoms: int = 51,
    ):
        super(CnnCategoricalValue, self).__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.conv, output_size = cnn((framestack, 16, 32))
        self.model = noisy_mlp(
            [output_size] + list(fc_layers),
            list(noisy_layers) + [self.action_dim * self.num_atoms],
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.conv(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.model(inp)
        inp = nn.functional.softmax(inp.view(-1, self.num_atoms)).view(
            -1, self.action_dim, self.num_atoms
        )
        return inp

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


dqn_value_registry = {
    "mlpdueling": MlpDuelingValue,
    "cnndueling": CnnDuelingValue,
    "mlpnoisy": MlpNoisyValue,
    "cnnnoisy": CnnNoisyValue,
    "mlpcategorical": MlpCategoricalValue,
    "cnncategorical": CnnCategoricalValue,
}


def get_dqn_value_from_name(name_: str):
    """
    Returns DQN model given the type of DQN

    :param name_: Name of the policy needed
    :type name_: str
    :returns: DQN class to be used
    """
    if name_ in dqn_value_registry:
        return dqn_value_registry[name_]
    raise NotImplementedError
