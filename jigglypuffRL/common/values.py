import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
import gym

# TODO (ajaysub110): add other value classes

class MlpValue(nn.Module):
    """
    Value object that implements actor critic, using a MLP (1 layers of 24)
    :param env: (Gym environment) The environment to learn from
    :param n_hidden: (int) number of neurons in hidden layers
    """
    def __init__(self, env, n_hidden=24):
        super(MlpValue, self).__init__()

        self.n_hidden = n_hidden
        self.state_space = None
        self.action_space = None

        # TODO (ajaysub110): add support for other space types
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_space = env.observation_space.shape[0]

        self.fc1 = nn.Linear(self.state_space, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        self.value_hist = Variable(torch.Tensor())
        self.loss_hist = Variable(torch.Tensor())

    def forward(self, x):
        model = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

        return model(x)

value_registry = {
    'MlpValue': MlpValue
}

def get_value_from_name(name):
    if name not in value_registry:
        raise ValueError("Error: unknown value type {}, the only ones available are {}".format(name, list(value_registry.keys)))
    return value_registry[name]