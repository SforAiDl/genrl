import numpy as np

from genrl.agents.bandits.multiarmed.base import MABAgent
from genrl.core.bandit import MultiArmedBandit


class EpsGreedyMABAgent(MABAgent):
    """
    Contextual Bandit Policy with Epsilon Greedy Action Selection Strategy.

    Refer to Section 2.3 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param eps: Probability with which a random action is to be selected.
    :type bandit: MultiArmedlBandit type object
    :type eps: float
    """

    def __init__(self, bandit: MultiArmedBandit, eps: float = 0.05):
        super(EpsGreedyMABAgent, self).__init__(bandit)
        self._eps = eps
        self._quality = np.zeros(shape=(bandit.bandits, bandit.arms))
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def eps(self) -> float:
        """float: Exploration constant"""
        return self._eps

    @property
    def quality(self) -> np.ndarray:
        """numpy.ndarray: Q values assigned by the policy to all actions"""
        return self._quality

    def select_action(self, context: int) -> int:
        """
        Select an action according to epsilon greedy startegy

        A random action is selected with espilon probability over
        the optimal action according to the current Q values to
        encourage exploration of the policy.

        :param context: the context to select action for
        :type context: int
        :returns: Selected action
        :rtype: int
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.quality[context])
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward recieved in this step.

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
        self.quality[context, action] += (reward - self.quality[context, action]) / (
            self.counts[context, action] + 1
        )
        self.counts[context, action] += 1
