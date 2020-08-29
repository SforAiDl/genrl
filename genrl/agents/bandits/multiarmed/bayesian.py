import numpy as np
from scipy import stats

from genrl.agents.bandits.multiarmed.base import MABAgent
from genrl.core.bandit import MultiArmedBandit


class BayesianUCBMABAgent(MABAgent):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: alpha value for beta distribution
    :param beta: beta values for beta distibution
    :param c: Confidence level which controls degree of exploration
    :type bandit: MultiArmedlBandit type object
    :type alpha: float
    :type beta: float
    :type c: float
    """

    def __init__(
        self,
        bandit: MultiArmedBandit,
        alpha: float = 1.0,
        beta: float = 1.0,
        confidence: float = 3.0,
    ):
        super(BayesianUCBMABAgent, self).__init__(bandit)
        self._c = confidence
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

    @property
    def confidence(self) -> float:
        """float: Confidence level which weights the exploration term"""
        return self._c

    def select_action(self, context: int) -> int:
        """
        Select an action according to bayesian upper confidence bound

        Take action that maximises a weighted sum of the Q values and
        a beta distribution paramerterized by alpha and beta
        and weighted by c for each action

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.quality[context]
            + stats.beta.std(self.a[context], self.b[context]) * self.confidence
        )
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward recieved in this step

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
