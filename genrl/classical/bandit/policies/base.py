from typing import List

import numpy as np

from ..bandits import Bandit


class BanditPolicy(object):
    """
    Base Class for Multi-armed Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of quality values is required
    :type bandit: Bandit type object
    """

    def __init__(self, bandit: Bandit, requires_init_run: bool = False):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []
        self._counts = np.zeros(self._bandit.arms)
        self._requires_init_run = requires_init_run

    @property
    def action_hist(self) -> List[int]:
        """
        Get the history of actions taken

        :returns: List of actions
        :rtype: list
        """
        return self._action_hist

    @property
    def regret_hist(self) -> List[float]:
        """
        Get the history of regrets computed for each step

        :returns: List of regrets
        :rtype: list
        """
        return self._regret_hist

    @property
    def regret(self) -> float:
        """
        Get the current regret

        :returns: The current regret
        :rtype: float
        """
        return self._regret

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received for each step

        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def counts(self) -> np.ndarray:
        """
        Get the number of times each action has been taken

        :returns: Numpy array with count for each action
        :rtype: numpy.ndarray
        """
        return self._counts

    def select_action(self, timestep: int) -> int:
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        raise NotImplementedError

    def learn(self, n_timesteps: int = 1000) -> None:
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received. If policy
        requires an initial run, it takes each action once before starting

        :param n_timesteps: number of steps to learn for
        :type: int
        """
        if self._requires_init_run:
            for action in range(self._bandit.arms):
                reward = self._bandit.step(action)
                self.update_params(action, reward)
            n_timesteps -= self._bandit.arms

        for timestep in range(n_timesteps):
            action = self.select_action(timestep)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
