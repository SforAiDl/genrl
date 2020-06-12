import numpy as np
from scipy import stats

from ..bandits import Bandit
from .base import BanditPolicy


class BayesianUCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: alpha value for beta distribution
    :param beta: beta values for beta distibution
    :param confidence: Confidence level which controls degree of exploration
    :type bandit: Bandit type object
    :type alpha: float
    :type beta: float
    :type confidence: float
    """

    def __init__(
        self,
        bandit: Bandit,
        alpha: float = 1.0,
        beta: float = 1.0,
        confidence: float = 3.0,
    ):
        super(BayesianUCBPolicy, self).__init__(bandit)
        self._c = confidence
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
        Get the beta value of beta distribution associated with the policy

        :returns: beta values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._b

    @property
    def c(self) -> float:
        """
        Get the confidence level which weights the exploration term

        :returns: Confidence level which controls degree of exploration
        :rtype: float
        """
        return self._c

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to bayesian upper confidence bound

        Take action that maximises a weighted sum of the quality values and
        a beta distribution paramerterized by alpha and beta
        and weighted by c for each action.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(self.quality + stats.beta.std(self.a, self.b) * self.c)
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
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1
