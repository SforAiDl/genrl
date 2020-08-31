from typing import Tuple

import numpy as np

from genrl.core.base import BasePolicy
from genrl.utils.utils import cnn, mlp


class MlpPolicy(BasePolicy):
    """
    MLP Policy

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type discrete: bool
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: Tuple = (32, 32),
        discrete: bool = True,
        *args,
        **kwargs
    ):
        super(MlpPolicy, self).__init__(
            state_dim, action_dim, hidden, discrete, **kwargs
        )
        self.activation = kwargs["activation"] if "activation" in kwargs else "relu"

        self.model = mlp(
            [state_dim] + list(hidden) + [action_dim],
            activation=self.activation,
            sac=self.sac,
        )


class CNNPolicy(BasePolicy):
    """
    CNN Policy

    :param framestack: Number of previous frames to stack together
    :param action_dim: Action dimensions of the environment
    :param fc_layers: Sizes of hidden layers
    :param discrete: True if action space is discrete, else False
    :param channels: Channel sizes for cnn layers
    :type framestack: int
    :type action_dim: int
    :type fc_layers: tuple or list
    :type discrete: bool
    :type channels: list or tuple
    """

    def __init__(
        self,
        framestack: int,
        action_dim: int,
        hidden: Tuple = (32, 32),
        discrete: bool = True,
        *args,
        **kwargs
    ):
        super(CNNPolicy, self).__init__(
            framestack, action_dim, hidden, discrete, **kwargs
        )
        channels = (framestack, 16, 32)

        self.conv, output_size = cnn(channels)

        self.fc = mlp([output_size] + list(hidden) + [action_dim], sac=self.sac)

    def forward(self, state: np.ndarray) -> np.ndarray:
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        action = self.fc(state)
        return action


policy_registry = {"mlp": MlpPolicy, "cnn": CNNPolicy}


def get_policy_from_name(name_: str):
    """
    Returns policy given the name of the policy

    :param name_: Name of the policy needed
    :type name_: str
    :returns: Policy Function to be used
    """
    if name_ in policy_registry:
        return policy_registry[name_]
    raise NotImplementedError
