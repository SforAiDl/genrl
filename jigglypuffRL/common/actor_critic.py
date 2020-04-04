import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.autograd import Variable
import gym
from jigglypuffRL.common import BasePolicy

# base classes
class BaseActor(nn.Module):
    def __init__(self, env):
        super(BaseActor, self).__init__()

        self.state_dim = None
        self.action_dim = None
        self.action_limit = None
        self.env = env

        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_dim = env.observation_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def sample_action(self, x):
        x = self.forward(x)
        c = self.action_distribution(x)
        action = c.sample()
        return action, c

    def action_distribution(self, x):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            x = nn.Softmax(dim=-1)(x)
            c = Categorical(probs=x)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            mean, log_std = x
            log_std = torch.clamp(log_std, -20, 2)
            c = Normal(mean, log_std.exp())
        else:
            raise NotImplementedError

        return c


class BaseCritic(nn.Module):
    def __init__(self, env):
        super(BaseCritic, self).__init__()

        self.state_dim = None
        self.action_dim = None

        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_dim = env.observation_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError

        self.value_hist = Variable(torch.Tensor())
        self.loss_hist = Variable(torch.Tensor())

    def forward(self):
        raise NotImplementedError


# Inherited classes
class ActorCritic(nn.Module):
    def __init__(self, env, network_type, layers, noise_std=0, det_stoc="det"):
        super(ActorCritic, self).__init__()

        self.actor, self.critic = get_actor_critic_from_name(network_type)
        self.actor = self.actor(env, layers)
        self.critic = self.critic(env, layers)
        self.det_stoc = det_stoc
        self.noise_std = noise_std

    def value(self, s, a):
        return self.critic.forward(s, a)

    def select_action(self, x):
        with torch.no_grad():
            if self.det_stoc == "det":
                return self.actor(x).cpu().numpy()
            elif self.det_stoc == "stoc":
                return self.actor.sample_action(x)
            else:
                raise ValueError


class MlpActor(BaseActor):
    def __init__(self, env, layers):
        super(MlpActor, self).__init__(env)

        self.action_limit = env.action_space.high[0]
        self.model = mlp([self.state_dim] + list(layers) + [self.action_dim])

    def forward(self, x):

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            x = nn.Softmax(dim=-1)(self.model(x))
        elif isinstance(self.env.action_space, gym.spaces.Box):
            x = nn.Tanh()(self.model(x))
            x = self.action_limit * x

        return x


class MlpCritic(BaseCritic):
    def __init__(self, env, layers):
        super(MlpCritic, self).__init__(env)

        self.model = mlp([self.state_dim + self.action_dim] + list(layers) + [1])

    def forward(self, s, a):
        return torch.squeeze(self.model(torch.cat([s, a], dim=-1)))


registry = {"Mlp": (MlpActor, MlpCritic)}


def get_actor_critic_from_name(name):
    if name not in registry:
        raise ValueError(
            "Error: unknown network type {}, the only ones available are {}".format(
                name, list(registry.keys)
            )
        )
    return registry[name]


# helper methods
def mlp(sizes):
    layers = []
    for j in range(len(sizes) - 1):  # 0, 1, 2
        act = nn.ReLU if j < len(sizes) - 2 else nn.Identity
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)
