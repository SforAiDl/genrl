import numpy as np

from genrl.classical.bandit.contextual_bandits import ContextualBandit
from genrl.classical.bandit.contextual_policies.base import CBPolicy


class GradientCBPolicy(CBPolicy):
    """
    Multi-Armed Bandit Solver with Softmax Action Selection Strategy.

    Refer to Section 2.8 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: The step size parameter for gradient based update
    :param temp: Temperature for softmax distribution over Q values of actions
    :type bandit: ContextualBandit type object
    :type alpha: float
    :type temp: float
    """

    def __init__(
        self, bandit: ContextualBandit, alpha: float = 0.1, temp: float = 0.01
    ):
        super(GradientCBPolicy, self).__init__(bandit)
        self._alpha = alpha
        self._temp = temp
        self._quality = np.zeros(shape=(bandit.bandits, bandit.arms))
        self._probability_hist = []

    @property
    def alpha(self) -> float:
        """
        Get the step size parameter for gradient based update of policy

        :returns: Step size which controls rate of learning for policy
        :rtype: float
        """
        return self._alpha

    @property
    def temp(self) -> float:
        """
        Get the temperature for softmax distribution over Q values of actions

        :returns: Temperature which controls softness of softmax distribution
        :rtype: float
        """
        return self._temp

    @property
    def quality(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._quality

    @property
    def probability_hist(self) -> np.ndarray:
        """
        Get the history of probabilty values assigned to each action for each timestep

        :returns: Numpy array of probability values for all actions
        :rtype: numpy.ndarray
        """
        return self._probability_hist

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        r"""
        Softmax with temperature
        :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i / temp)}{\sum_j \exp(x_j / temp)}`

        :param x: Set of values to compute softmax over
        :type x: numpy.ndarray
        :returns: Computed softmax over given values
        :rtype: numpy.ndarray
        """
        exp = np.exp(x / self.temp)
        total = np.sum(exp)
        p = exp / total
        return p

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according by softmax action selection strategy

        Action is sampled from softmax distribution computed over
        the Q values for all actions

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        probabilities = self._softmax(self.quality[context])
        action = np.random.choice(self._bandit.arms, 1, p=probabilities)[0]
        self.action_hist.append((context, action))
        self.probability_hist.append(probabilities)
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and that
        of the action. Updates the Q values through a gradient ascent step

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.quality[context]) - self.quality[context, action]
        self.regret_hist.append(self.regret)

        # compute reward baseline by taking mean of all rewards till t-1
        if len(self.reward_hist) <= 1:
            reward_baseline = 0.0
        else:
            reward_baseline = np.mean(self.reward_hist[:-1])

        current_probailities = self.probability_hist[-1]

        # update Q values for the action taken and those not taken seperately
        self.quality[context, action] += (
            self.alpha * (reward - reward_baseline) * (1 - current_probailities[action])
        )
        actions_not_taken = np.arange(self._bandit.arms) != action
        self.quality[context, actions_not_taken] += (
            -1
            * self.alpha
            * (reward - reward_baseline)
            * current_probailities[actions_not_taken]
        )
