import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def noisy_mlp(fc_layers, noisy_layers):
    """
    generate Noisy MLP model given sizes of each layer
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
    def __init__(self, state_dim, action_dim, hidden=(128, 128)):
        super(DuelingDQNValueMlp, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(self.state_dim, hidden[0]),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], self.action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1)
        )

    def forward(self, state):
        features = self.feature(state)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(
            out_features, in_features
        ))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(
            out_features, in_features
        ))
        self.register_buffer('weight_epsilon', torch.FloatTensor(
            out_features, in_features
        ))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, state):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(
                Variable(self.weight_epsilon)
            )
            bias = self.bias_mu + self.bias_sigma.mul(
                Variable(self.bias_epsilon)
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(state, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.bias_sigma.size(0))
        )

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQNValue(nn.Module):
    def __init__(
        self, state_dim, action_dim, fc_layers=(128), noisy_layers=(128, 128)
    ):
        super(NoisyDQNValue, self).__init__()

        self.noisy = noisy_mlp(
            [state_dim] + list(fc_layers),
            list(noisy_layers) + [action_dim]
        )

    def forward(self, state):
        return self.noisy(state)

    def reset_noise(self):
        for layer in self.noisy:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class CategoricalDQNValue(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        num_atoms,
        Vmin,
        Vmax,
        fc_layers=(128, 128),
        noisy_layers=(128, 512)
    ):
        super(CategoricalDQNValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.model = noisy_mlp(
            [state_dim] + list(fc_layers),
            list(noisy_layers) + [self.num_actions * self.num_atoms]
        )

    def forward(self, state):
        features = self.model(state)
        dist = F.softmax(features.view(-1, self.num_atoms)).view(
            -1, self.num_actions, self.num_atoms
        )
        return dist

    def reset_noise(self):
        for layer in self.noisy:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
