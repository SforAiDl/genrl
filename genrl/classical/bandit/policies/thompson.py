import numpy as np

from ...bandit import Bandit
from .base import BanditPolicy


class ThompsonSamplingPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    :param bandit: The Bandit to solve
    :param a: alpha value for beta distribution
    :param b: beta values for beta distibution
    :type bandit: Bandit type object
    :type a: float
    :type b: float
    """

    def __init__(self, bandit: Bandit, alpha: float = 1.0, beta: float = 1.0):
        super(ThompsonSamplingPolicy, self).__init__(bandit)
        self._a = alpha * np.ones(self._bandit.arms)
        self._b = beta * np.ones(self._bandit.arms)

    @property
    def quality(self) -> np.ndarray:
        """
        Compute the q values for all the actions for alpha, beta and c

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self.a / (self.a + self.b)

    @property
    def a(self) -> np.ndarray:
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._a

    @property
    def b(self) -> np.ndarray:
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._b

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to Thompson Sampling

        Samples are taken from beta distribution parameterized by
        alpha and beta for each action. The action with the highest
        sample is selected.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        sample = np.random.beta(self.a, self.b)
        action = np.argmax(sample)
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
        that of the action. Updates the alpha value of beta distribution
        by adding the reward while the beta value is updated by adding
        1 - reward. Update the counts the action taken.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: int
        """
        self.reward_hist.append(reward)
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1
