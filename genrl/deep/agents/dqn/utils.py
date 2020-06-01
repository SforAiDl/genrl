import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from genrl.deep.common.utils import mlp, cnn
from typing import List, Tuple


def noisy_mlp(fc_layers: List[int], noisy_layers: List[int]):
    """
    Generate Noisy MLP model given sizes of each fully connected and noisy layers

    :param fc_layers: list of fc layers
    :param noisy_layers: list of noisy layers 
    :type fc_layers: list
    :type noisy_layers: list
    :returns: Noisy model
    :rtype: Pytorch model
    """
    model = []
    for j in range(len(fc_layers) - 1):
        model += [nn.Linear(fc_layers[j], fc_layers[j + 1]), nn.ReLU()]
    for j in range(len(noisy_layers) - 1):
        model += [NoisyLinear(noisy_layers[j], noisy_layers[j + 1])]
        if j < len(noisy_layers) - 2:
            model += [nn.ReLU()]
    return nn.Sequential(*model)


class DuelingDQNValueMlp(nn.Module):
    """
    Class for Dueling DQN's MLP Value function 

    :param state_dim: Observation space
    :param action_dim: Action space
    :param hidden: Number of hidden Nodes
    :type state_dim: int, float, ...
    :type action_dim: int, float, ...
    :type hidden: tuple 
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple = (128, 128)):
        super(DuelingDQNValueMlp, self).__init__()

        self.feature = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU())

        self.advantage = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean()


class DuelingDQNValueCNN(nn.Module):
    """
    Class for Dueling DQN's CNN Value function 

    :param action_dim: Action space
    :param framestack: Number of frames you're considering in your history
    :param fc_layers: no of units in fc layers 
    :type action_dim: int, float, ...
    :type framestack: int
    :type fc_layers: tuple
    """

    def __init__(self, action_dim: int, framestack: int = 4, fc_layers: Tuple = (256,)):
        super(DuelingDQNValueCNN, self).__init__()

        self.action_dim = action_dim

        self.conv, output_size = cnn((framestack, 16, 32))

        self.advantage = mlp([output_size] + list(fc_layers) + [action_dim])
        self.value = mlp([output_size] + list(fc_layers) + [1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class NoisyLinear(nn.Module):
    """
    Noisy Layer definition for NoisyDQN

    :param in_features: input features for the network
    :param out_features: output features for the network
    :param std_init: Used for initializing weights 
    :type in_features: int
    :type out_features: int
    :type std_init: float 
    """

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
            weight = self.weight_mu + self.weight_sigma.mul(
                Variable(self.weight_epsilon)
            )
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(state, weight, bias)

    def reset_parameters(self) -> None:
        """
        Reset the parameters 
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self) -> None:
        """
        Reset Noisy layers weights
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQNValue(nn.Module):
    """
    Class for Dueling DQN's MLP Value function

    :param state_dim: Observation space
    :param action_dim: Action space
    :param fc_layers: no of units in fc layers
    :param noisy_layers: no of units in noisy layers
    :type state_dim: int, float, ... 
    :type action_dim: int, float, ...
    :type fc_layers: tuple  
    :type noisy_layers: tuple
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        fc_layers: Tuple = (128,),
        noisy_layers: Tuple = (128, 128),
    ):
        super(NoisyDQNValue, self).__init__()

        self.model = noisy_mlp(
            [state_dim] + list(fc_layers), list(noisy_layers) + [action_dim]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class NoisyDQNValueCNN(nn.Module):
    """
    Class for Dueling DQN's CNN Value function

    :param action_dim: Action space
    :param framestack: Number of frames you're considering in your history
    :param fc_layers: no of units in fc layers
    :param noisy_layers: no of units in noisy layers
    :type action_dim: int, float, ...
    :type framestack: int
    :type fc_layers: tuple  
    :type noisy_layers: tuple
    """

    def __init__(
        self,
        action_dim: int,
        framestack: int = 4,
        fc_layers: Tuple = (128,),
        noisy_layers: Tuple = (128, 128),
    ):
        super(NoisyDQNValueCNN, self).__init__()

        self.conv, output_size = cnn((framestack, 16, 32))

        self.model = noisy_mlp(
            [output_size] + list(fc_layers), list(noisy_layers) + [action_dim]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CategoricalDQNValue(nn.Module):
    """
    Class for Categorical DQN's MLP Value function

    :param state_dim: Observation space
    :param action_dim: Action space
    :param num_atoms: Number of atoms in the Categorical network
    :param fc_layers: no of units in fc layers
    :param noisy_layers: no of units in noisy layers
    :type state_dim: int, float, ... 
    :type action_dim: int, float, ...
    :type num_atoms: int
    :type fc_layers: tuple  
    :type noisy_layers: tuple
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_atoms: int,
        fc_layers: Tuple = (128, 128),
        noisy_layers: Tuple = (128, 512),
    ):
        super(CategoricalDQNValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.model = noisy_mlp(
            [state_dim] + list(fc_layers),
            list(noisy_layers) + [self.action_dim * self.num_atoms],
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.model(state)
        dist = F.softmax(features.view(-1, self.num_atoms)).view(
            -1, self.action_dim, self.num_atoms
        )
        return dist

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CategoricalDQNValueCNN(nn.Module):
    """
    Class for Categorical DQN's MLP Value function

    :param action_dim: Action space
    :param num_atoms: Number of atoms in the Categorical network
    :param framestack: Number of frames you're considering in your history
    :param fc_layers: no of units in fc layers
    :param noisy_layers: no of units in noisy layers
    :type action_dim: int, float, ...
    :type num_atoms: int
    :type framestack: int
    :type fc_layers: tuple  
    :type noisy_layers: tuple
    """

    def __init__(
        self,
        action_dim: int,
        num_atoms: int,
        framestack: int = 4,
        fc_layers: Tuple = (128, 128),
        noisy_layers: Tuple = (128, 512),
    ):
        super(CategoricalDQNValueCNN, self).__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.conv, output_size = cnn((framestack, 16, 32))
        self.model = noisy_mlp(
            [output_size] + list(fc_layers),
            list(noisy_layers) + [self.action_dim * self.num_atoms],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = F.softmax(x.view(-1, self.num_atoms)).view(
            -1, self.action_dim, self.num_atoms
        )
        return x

    def reset_noise(self) -> None:
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
