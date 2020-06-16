import numpy as np

from ...bandit import Bandit
from .base import BanditPolicy


class UCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound based
    Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param c: Confidence level which controls degree of exploration
    :type bandit: Bandit type object
    :type c: float
    """

    def __init__(self, bandit: Bandit, confidence: float = 1.0):
        super(UCBPolicy, self).__init__(bandit, requires_init_run=True)
        self._c = confidence
        self._quality = np.zeros(bandit.arms)

    @property
    def confidence(self) -> float:
        """
        Get the confidence level which weights the exploration term

        :returns: Confidence level which controls degree of exploration
        :rtype: float
        """
        return self._c

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
        Select an action according to upper confidence bound action selction

        Take action that maximises a weighted sum of the quality values for the action
        and an exploration encouragement term controlled by c.

        :param timestep: timestep to choose action for
        :type timesteps: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.quality
            + self.confidence * np.sqrt(2 * np.log(timestep + 1) / (self.counts + 1))
        )
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
