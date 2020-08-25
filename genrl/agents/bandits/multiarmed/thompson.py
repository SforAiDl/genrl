import numpy as np

from genrl.agents.bandits.multiarmed.base import MABAgent
from genrl.core.bandit import MultiArmedBandit


class ThompsonSamplingMABAgent(MABAgent):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    :param bandit: The Bandit to solve
    :param a: alpha value for beta distribution
    :param b: beta values for beta distibution
    :type bandit: MultiArmedlBandit type object
    :type a: float
    :type b: float
    """

    def __init__(self, bandit: MultiArmedBandit, alpha: float = 1.0, beta: float = 1.0):
        super(ThompsonSamplingMABAgent, self).__init__(bandit)
        self._a = alpha * np.ones(shape=(bandit.bandits, bandit.arms))
        self._b = beta * np.ones(shape=(bandit.bandits, bandit.arms))

    @property
    def quality(self) -> np.ndarray:
        """numpy.ndarray: Q values for all the actions for alpha, beta and c"""
        return self.a / (self.a + self.b)

    @property
    def a(self) -> np.ndarray:
        """numpy.ndarray: alpha parameter of beta distribution associated with the policy"""
        return self._a

    @property
    def b(self) -> np.ndarray:
        """numpy.ndarray: beta parameter of beta distribution associated with the policy"""
        return self._b

    def select_action(self, context: int) -> int:
        """
        Select an action according to Thompson Sampling

        Samples are taken from beta distribution parameterized by
        alpha and beta for each action. The action with the highest
        sample is selected.

        :param context: the context to select action for
        :type context: int
        :returns: Selected action
        :rtype: int
        """
        sample = np.random.beta(self.a[context], self.b[context])
        action = np.argmax(sample)
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the alpha value of beta distribution
        by adding the reward while the beta value is updated by adding
        1 - reward. Update the counts the action taken.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self.a[context, action] += reward
        self.b[context, action] += 1 - reward
        self._regret += max(self.quality[context]) - self.quality[context, action]
        self.regret_hist.append(self.regret)
        self.counts[context, action] += 1
