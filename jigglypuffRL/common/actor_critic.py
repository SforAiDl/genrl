import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.autograd import Variable
import gym
from jigglypuffRL.common import BasePolicy


class BaseActor(nn.Module):
    def __init__(self, env):
        super(BaseActor, self).__init__(env)

        self.state_space = None
        self.action_space = None
        self.policy = None
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

    def forward(self):
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
        super(BaseCritic, self).__init__(env)

        self.n_hidden = n_hidden
        self.state_space = None
        self.action_space = None

        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_space = env.observation_space.shape[0]

        self.value_hist = Variable(torch.Tensor())
        self.loss_hist = Variable(torch.Tensor())

    def forward(self):
        raise NotImplementedError


class ActorCritic(nn.Module):
    def __init__(self, env, Actor, Critic):
        super(ActorCritic).__init__(env)

        self.actor = Actor(env)
        self.critic = Critic(env)

    def value(self, x):
        return self.critic.forward(x)

    def sample_action(self, x):
        return self.actor.sample_action(x)
