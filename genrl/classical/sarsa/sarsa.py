from typing import Tuple

import gym
import numpy as np


class SARSA:
    """
    SARSA Algorithm.

    Paper- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf

    Attributes:
        env (gym.Env): Environment with which agent interacts.
        epsilon (float, optional): exploration coefficient for epsilon-greedy exploration.
        gamma (float, optional): discount factor.
        lr (float, optional): learning rate for optimizer.
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

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.e = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Epsilon greedy selection of epsilon in the explore phase.

        Args:
            state (np.ndarray): Environment state.
            explore (bool, optional): True if exploration is required. False if not.

        Returns:
            np.ndarray: action.
        """
        if explore:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[state, :])

    def update(self, transition: Tuple) -> None:
        """Update the Q table and e values

        Args:
            transition (Tuple): transition 4-tuple used to update Q-table.
                In the form (state, action, reward, next_state)
        """
        state, action, reward, next_state = transition

        next_action = self.get_action(next_state)
        delta = reward + self.gamma * (
            self.Q[next_state, next_action] - self.Q[state, action]
        )
        self.e[state, action] += 1

        for _si in range(self.env.observation_space.n):
            for _ai in range(self.env.action_space.n):
                self.Q[state, action] += self.lr * (delta * self.e[state, action])
                self.e[state, action] = self.gamma * (
                    self.lmbda * self.e[state, action]
                )
