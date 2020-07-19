from typing import Tuple

import torch
from torch import nn as nn

from ...common import cnn, mlp
from .base import DQN
from .utils import noisy_mlp


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


dqn_value_registry = {
    "mlpdueling": MlpDuelingValue,
    "mlpnoisy": MlpNoisyValue,
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
