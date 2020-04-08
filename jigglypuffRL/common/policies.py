import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.autograd import Variable
import gym

# TODO (ajaysub110): add other policy classes


class BasePolicy(nn.Module):
    """
    Base policy that all other policies should inherit.
    :param env: (Gym environment) The environment to learn from
    """

    def __init__(self, env):
        super(BasePolicy, self).__init__()

        self.state_space = None
        self.action_space = None
        self.env = env

        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_space = env.observation_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space = env.action_space.shape[0]
        else:
            raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def sample_action(self, x):
        x = self.forward(x)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            x = nn.Softmax(dim=-1)(x)
            c = Categorical(probs=x)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            mean, log_std = x
            log_std = torch.clamp(log_std, -20, 2)
            c = Normal(mean, log_std.exp())
        else:
            raise NotImplementedError

        action = c.sample()
        return action, c


class MlpPolicy(BasePolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 24)
    :param env: (Gym environment) The environment to learn from
    :param n_hidden: (int) number of neurons in hidden layers
    """

    def __init__(self, env, n_hidden=16):
        super(MlpPolicy, self).__init__(env)

        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(self.state_space, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, self.action_space)

        if isinstance(env.action_space, gym.spaces.Box):
            self.fc4 = nn.Linear(n_hidden, self.action_space)

        self.policy_hist = Variable(torch.Tensor())
        self.traj_reward = []
        self.loss_hist = Variable(torch.Tensor())

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        y = self.fc3(x)

        if isinstance(self.env.action_space, gym.spaces.Box):
            mean = y
            log_std = self.fc4(x)
            return (mean, log_std)
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            return y


policy_registry = {"MlpPolicy": MlpPolicy}


def get_policy_from_name(name):
    if name not in policy_registry:
        raise ValueError(
            "Error: unknown policy type {}, the only ones available are {}".format(
                name, list(policy_registry.keys)
            )
        )
    return policy_registry[name]
