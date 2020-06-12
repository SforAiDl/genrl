from typing import Any, Dict, List

import numpy as np
from scipy import stats

from ..bandits import Bandit
from .base import BanditPolicy


class EpsGreedyPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Epsilon Greedy Action Selection Strategy.

    Refer to Section 2.3 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param eps: Probability with which a random action is to be selected.
    :type bandit: Bandit type object
    :type eps: float
    """

    def __init__(self, bandit: Bandit, eps: float = 0.05):
        super(EpsGreedyPolicy, self).__init__(bandit)
        self._eps = eps
        self._quality = np.zeros(bandit.arms)

    @property
    def eps(self) -> float:
        """
        Get the asscoiated epsilon for the policy

        :returns: Probability with which a random action is to be selected
        :rtype: float
        """
        return self._eps

    @property
    def quality(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._quality

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to epsilon greedy startegy

        A random action is selected with espilon probability over
        the optimal action according to the current quality values to
        encourage exploration of the policy.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.quality)
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
        that of the action. Updates the quality values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.quality[action] += (reward - self.quality[action]) / (
            self.counts[action] + 1
        )
        self.counts[action] += 1
