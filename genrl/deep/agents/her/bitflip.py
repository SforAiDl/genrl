import random
from collections import namedtuple
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

class BitFlipEnvironment:

    def __init__(self, bits):
        self.action_space = spaces.Discrete(bits)
        self.max_steps=bits
        self.observation_space = spaces.MultiBinary(bits)
        self.goal_space = spaces.MultiBinary(bits)
        self.bits = bits
        self.state = torch.zeros((self.bits, ))
        self.goal = torch.zeros((self.bits, ))
        self.reset()

    def reset(self):
        self.state = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        self.goal = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        if torch.equal(self.state, self.goal):
            self.reset()
        return self.state.clone(), self.goal.clone()

    def step(self, action):
        self.state[action] = 1 - self.state[action]  # Flip the bit on position of the action
        reward, done = self.compute_reward(self.state, self.goal)
        return self.state.clone(), reward, done

    def render(self):
        print("State: {}".format(self.state.tolist()))
        print("Goal : {}\n".format(self.goal.tolist()))

    @staticmethod
    def compute_reward(state, goal):
        done = torch.equal(state, goal)
        return torch.tensor(0.0 if done else -1.0), done