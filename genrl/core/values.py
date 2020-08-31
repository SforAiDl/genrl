from typing import Tuple, Type, Union

import numpy as np
import torch
from torch.nn import functional as F

from genrl.core.base import BaseValue
from genrl.core.noise import NoisyLinear
from genrl.utils.utils import cnn, mlp, noisy_mlp


def _get_val_model(
    arch: str,
    val_type: str,
    state_dim: str,
    fc_layers: Tuple,
    action_dim: int = None,
    activation: str = "relu",
):
    """
        Returns Neural Network given specifications

        :param arch: Specifies type of architecture "mlp" for MLP layers
        :param val_type: Specifies type of value function: (
    "V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a))
        :param state_dim: State dimensions of environment
        :param action_dim: Action dimensions of environment
        :param hidden: Sizes of hidden layers
        :type arch: string
        :type val_type: string
        :type state_dim: string
        :type action_dim: int
        :type hidden: tuple or list
        :returns: Neural Network model to be used for the Value function
    """
    if val_type == "V":
        return arch([state_dim, *fc_layers, 1], activation=activation)
    elif val_type == "Qsa":
        return arch([state_dim + action_dim, *fc_layers, 1], activation=activation)
    elif val_type == "Qs":
        return arch([state_dim, *fc_layers, action_dim], activation=activation)
    else:
        raise ValueError


class MlpValue(BaseValue):
    """
        MLP Value Function class

        :param state_dim: State dimensions of environment
        :param action_dim: Action dimensions of environment
        :param val_type: Specifies type of value function: (
    "V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a))
        :param hidden: Sizes of hidden layers
        :type state_dim: int
        :type action_dim: int
        :type val_type: string
        :type hidden: tuple or list
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = None,
        val_type: str = "V",
        fc_layers: Tuple = (32, 32),
        **kwargs,
    ):
        super(MlpValue, self).__init__(state_dim, action_dim)
        self.val_type = val_type
        self.fc_layers = fc_layers

        self.activation = kwargs["activation"] if "activation" in kwargs else "relu"

        self.model = _get_val_model(
            mlp, val_type, state_dim, fc_layers, action_dim, self.activation
        )


class CnnValue(MlpValue):
    """
        CNN Value Function class

        :param framestack: Number of previous frames to stack together
        :param action_dim: Action dimension of environment
        :param val_type: Specifies type of value function: (
    "V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a))
        :param fc_layers: Sizes of hidden layers
        :type framestack: int
        :type action_dim: int
        :type val_type: string
        :type fc_layers: tuple or list
    """

    def __init__(self, *args, **kwargs):
        super(CnnValue, self).__init__(*args, **kwargs)

        self.conv, self.output_size = cnn(
            (self.state_dim, 16, 32), activation=self.activation
        )
        self.model = mlp([self.output_size, *self.fc_layers, self.action_dim])

    def _cnn_forward(self, state):
        batch_size, n_envs, frame, H, B = state.shape
        state = self.conv(state.view(-1, frame, H, B))
        return state.view(batch_size, n_envs, -1)

    def forward(self, state: np.ndarray) -> np.ndarray:
        value = self._cnn_forward(state)
        return self.model.forward(value)


class MlpDuelingValue(MlpValue):
    """Class for Dueling DQN's MLP Q-Value function

    Attributes:
        state_dim (int): Observation space dimensions
        action_dim (int): Action space dimensions
        hidden (:obj:`tuple`): Hidden layer dimensions
    """

    def __init__(self, *args, **kwargs):
        super(MlpDuelingValue, self).__init__(*args, **kwargs)
        self.feature = mlp([self.state_dim, *self.fc_layers[:-1]])
        self.advantage = mlp([self.fc_layers[-1], self.action_dim])
        self.value = mlp([self.fc_layers[-1], 1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.feature(state))
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean()


class CnnDuelingValue(CnnValue):
    """Class for Dueling DQN's MLP Q-Value function

    Attributes:
        framestack (int): No. of frames being passed into the Q-value function
        action_dim (int): Action space dimensions
        fc_layers (:obj:`tuple`): Hidden layer dimensions
    """

    def __init__(self, *args, **kwargs):
        super(CnnDuelingValue, self).__init__(*args, **kwargs)
        self.advantage = mlp([self.output_size, *self.fc_layers, self.action_dim])
        self.value = mlp([self.output_size, *self.fc_layers, 1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self._cnn_forward(inp)
        advantage = self.advantage(inp)
        value = self.value(inp)
        return value + advantage - advantage.mean()


class MlpNoisyValue(MlpValue):
    def __init__(self, *args, noisy_layers: Tuple = (128, 512), **kwargs):
        super(MlpNoisyValue, self).__init__(*args, **kwargs)

        self.noisy_layers = noisy_layers
        self.num_atoms = kwargs["num_atoms"] if "num_atoms" in kwargs else 1

        self.model = noisy_mlp(
            [self.state_dim, *self.fc_layers],
            [*self.noisy_layers, self.action_dim * self.num_atoms],
            self.activation,
        )

    def reset_noise(self) -> None:
        """
        Resets noise for any Noisy layers in Value function
        """
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CnnNoisyValue(CnnValue, MlpNoisyValue):
    """Class for Noisy DQN's CNN Q-Value function

    Attributes:
        state_dim (int): Number of previous frames to stack together
        action_dim (int): Action space dimensions
        fc_layers (:obj:`tuple`): Fully connected layer dimensions
        noisy_layers (:obj:`tuple`): Noisy layer dimensions
        num_atoms (int): Number of atoms used to discretise the
            Categorical DQN value distribution
    """

    def __init__(self, *args, **kwargs):
        super(CnnNoisyValue, self).__init__(*args, **kwargs)

        self.model = noisy_mlp(
            [self.output_size, *self.fc_layers],
            [*self.noisy_layers, self.action_dim * self.num_atoms],
            self.activation,
        )

    def forward(self, state: np.ndarray) -> np.ndarray:
        value = self._cnn_forward(state)
        return self.model.forward(value)


class MlpCategoricalValue(MlpNoisyValue):
    """Class for Categorical DQN's MLP Q-Value function

    Attributes:
        state_dim (int): Observation space dimensions
        action_dim (int): Action space dimensions
        fc_layers (:obj:`tuple`): Fully connected layer dimensions
        noisy_layers (:obj:`tuple`): Noisy layer dimensions
        num_atoms (int): Number of atoms used to discretise the
            Categorical DQN value distribution
    """

    def __init__(self, *args, **kwargs):
        super(MlpCategoricalValue, self).__init__(*args, **kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size, n_envs, _ = state.shape
        features = self.model(state)
        return F.softmax(features.view(-1, self.num_atoms), dim=0).view(
            batch_size, n_envs, self.action_dim, self.num_atoms
        )


class CnnCategoricalValue(CnnNoisyValue):
    """Class for Categorical DQN's CNN Q-Value function

    Attributes:
        framestack (int): No. of frames being passed into the Q-value function
        action_dim (int): Action space dimensions
        fc_layers (:obj:`tuple`): Fully connected layer dimensions
        noisy_layers (:obj:`tuple`): Noisy layer dimensions
        num_atoms (int): Number of atoms used to discretise the
            Categorical DQN value distribution
    """

    def __init__(self, *args, **kwargs):
        super(CnnCategoricalValue, self).__init__(*args, **kwargs)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self._cnn_forward(state)
        features = self.model(features)
        batch_size, env, _ = features.shape
        return F.softmax(features.view(-1, self.num_atoms), dim=0).view(
            batch_size, env, self.action_dim, self.num_atoms
        )


value_registry = {
    "mlp": MlpValue,
    "cnn": CnnValue,
    "mlpdueling": MlpDuelingValue,
    "cnndueling": CnnDuelingValue,
    "mlpnoisy": MlpNoisyValue,
    "cnnnoisy": CnnNoisyValue,
    "mlpcategorical": MlpCategoricalValue,
    "cnncategorical": CnnCategoricalValue,
}


def get_value_from_name(name_: str) -> Union[Type[MlpValue], Type[CnnValue]]:
    """
    Gets the value function given the name of the value function

    :param name_: Name of the value function needed
    :type name_: string
    :returns: Value function
    """
    if name_ in value_registry:
        return value_registry[name_]
    raise NotImplementedError
