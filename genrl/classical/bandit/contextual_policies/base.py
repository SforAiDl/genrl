from typing import List, Tuple

import numpy as np

from genrl.classical.bandit.contextual_bandits import ContextualBandit


class CBPolicy(object):
    """
    Base Class for Contextual Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of Q values is required
    :type bandit: Bandit type object
    """

    def __init__(self, bandit: ContextualBandit):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def action_hist(self) -> Tuple[int, int]:
        """
        Get the history of actions taken for contexts

        :returns: List of context, actions pairs
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

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
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
        context = self._bandit.reset()
        for t in range(n_timesteps):
            action = self.select_action(context, t)
            context, reward = self._bandit.step(action)
            self.update_params(context, action, reward)
