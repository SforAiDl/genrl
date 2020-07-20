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


dqn_value_registry = {
    "mlpdueling": MlpDuelingValue,
    "mlpnoisy": MlpNoisyValue,
    "mlpcategorical": MlpCategoricalValue,
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
