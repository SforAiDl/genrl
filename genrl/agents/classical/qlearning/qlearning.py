from typing import Any, Dict, Tuple

import gym
import numpy as np


class QLearning:
    """Q-Learning Algorithm.

    Paper- https://link.springer.com/article/10.1007/BF00992698

    Attributes:
        env (gym.Env): Environment with which agent interacts.
        epsilon (float, optional): exploration coefficient for epsilon-greedy exploration.
        gamma (float, optional): discount factor.
        lr (float, optional): learning rate for optimizer.
    """

    def __init__(
        self, env: gym.Env, epsilon: float = 0.9, gamma: float = 0.95, lr: float = 0.01
    ):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        self.mean_reward = None

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

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
        """Update the Q table.

        Args:
            transition (Tuple): transition 4-tuple used to update Q-table.
                In the form (state, action, reward, next_state)
        """
        state, action, reward, next_state = transition

        self.Q[state, action] += self.lr * (
            reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {"epsilon": self.epsilon, "gamma": self.gamma, "lr": self.lr}

        return hyperparams
