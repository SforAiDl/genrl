import numpy as np
import gym
from typing import Tuple


class SARSA:
    """
    SARSA Algorithm

    Paper- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf

    :param env: standard gym environment to train on
    :param epsilon: (float) exploration coefficient
    :param lmbda: (float) eligibility trace coefficient
    :param gamma: (float) discount factor
    :param lr: (float) learning rate
    :type env: Gym environment
    :type epsilon: float
    :type lmbda: float
    :type gamma: float
    :type lr: float
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 0.9,
        lmbda: float = 0.9,
        gamma: float = 0.95,
        lr: float = 0.01,
    ):
        self.env = env
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros(
            (self.env.observation_space.n,
             self.env.action_space.n))

        self.e = np.zeros(
            (self.env.observation_space.n,
             self.env.action_space.n))

    def get_action(
            self,
            state: np.ndarray,
            explore: bool = True) -> np.ndarray:
        """
        Epsilon greedy selection of epsilon in the explore phase

        :param s: Current state
        :param explore: Whether you are exploring or exploiting
        :type s: int, float, ...
        :type explore: bool
        :returns: Action based on the Q table
        :rtype: int, float, ...
        """
        if explore:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[state, :])

    def update(self, transition: Tuple) -> None:
        """
        Update the Q table and e values

        :param transition: step taken in the envrionment
        """
        state, action, reward, next_state = transition

        next_action = self.get_action(next_state)
        delta = reward + self.gamma * (
            self.Q[next_state, next_action] - self.Q[state, action])
        self.e[state, action] += 1

        for _si in range(self.env.observation_space.n):
            for _ai in range(self.env.action_space.n):
                self.Q[state, action] += self.lr * (
                    delta * self.e[state, action])
                self.e[state, action] = self.gamma * (
                    self.lmbda * self.e[state, action])
