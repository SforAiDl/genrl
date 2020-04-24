import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from jigglypuffRL.common.base import BaseValue
from jigglypuffRL.common.utils import mlp, cnn


class DuelingDQNValueMlp(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNValueMlp, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature = nn.Sequential(nn.Linear(self.state_dim, 128), nn.ReLU())

        self.advantage = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.action_dim)
        )

        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()


class DuelingDQNValueCNN(nn.Module):
    def __init__(self, action_dim, history_length=4, fc_layers=(256,)):
        super(DuelingDQNValueCNN, self).__init__()

        self.action_dim = action_dim

        self.conv, output_size = cnn((history_length, 16, 32))

        self.advantage = mlp([output_size] + list(fc_layers) + [action_dim])
        self.value = mlp([output_size] + list(fc_layers) + [1])

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQNValue(nn.Module):
    def __init__(self, num_states, num_actions):
        super(NoisyDQNValue, self).__init__()
        
        self.linear =  nn.Linear(num_states, 128)
        self.noisy_layer_1 = NoisyLinear(128, 128)
        self.noisy_layer_2 = NoisyLinear(128, num_actions)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy_layer_1(x))
        x = self.noisy_layer_2(x)
        return x
    
    def reset_noise(self):
        self.noisy_layer_1.reset_noise()
        self.noisy_layer_2.reset_noise()


class NoisyDQNValueCNN(nn.Module):
    def __init__(self, action_dim, history_length=4, fc_layers=(256,)):
        super(NoisyDQNValueCNN, self).__init__()

        self.conv, output_size = cnn((history_length, 16, 32))

        self.fc = nn.Linear(output_size, 128)
        self.noisy_layer_1 = NoisyLinear(128, 128)
        self.noisy_layer_2 = NoisyLinear(128, action_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.noisy_layer_1(x))
        x = self.noisy_layer_2(x)
        return x

    def reset_noise(self):
        self.noisy_layer_1.reset_noise()
        self.noisy_layer_2.reset_noise()


class CategoricalDQNValue(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(CategoricalDQNValue, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.noisy1 = NoisyLinear(128, 512)
        self.noisy2 = NoisyLinear(512, self.num_actions * self.num_atoms)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return x
        
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class CategoricalDQNValueCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax, history_length=4, fc_layers=(128, 128)):
        super(CategoricalDQNValueCNN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.conv, output_size = cnn((history_length, 16, 32))
        self.fc = mlp([output_size] + list(fc_layers))
        self.noisy1 = NoisyLinear(fc_layers[-1], 512)
        self.noisy2 = NoisyLinear(512, self.num_actions*self.num_atoms)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
