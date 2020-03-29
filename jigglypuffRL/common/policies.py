import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
import gym

# TODO (ajaysub110): add other policy classes

class MlpPolicy(nn.Module):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 24)
    :param env: (Gym environment) The environment to learn from
    :param n_hidden: (int) number of neurons in hidden layers
    """
    def __init__(self, env, n_hidden=24):
        super(MlpPolicy, self).__init__()

        self.n_hidden = n_hidden
        self.state_space = None
        self.action_space = None

        # TODO (ajaysub110): add support for other space types
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_space = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.state_space, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, self.action_space)

        self.policy_hist = Variable(torch.Tensor())
        self.traj_reward = []
        self.loss_hist = Variable(torch.Tensor())

    def forward(self, x):
        model = nn.Sequential(
            self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3,
            nn.Softmax(dim=-1)
        )

        return model(x)

policy_registry = {
    'MlpPolicy': MlpPolicy
}

def get_policy_from_name(name):
    if name not in policy_registry:
        raise ValueError("Error: unknown policy type {}, the only ones available are {}".format(name, list(policy_registry.keys)))
    return policy_registry[name]