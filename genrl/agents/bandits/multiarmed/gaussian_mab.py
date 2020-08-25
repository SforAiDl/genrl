from typing import Tuple

import numpy as np

from genrl.core.bandit import MultiArmedBandit


class GaussianMAB(MultiArmedBandit):
    """
    Contextual Bandit with categorial context and gaussian reward distribution

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param reward_means: Mean of gaussian distribution for each reward
    :type bandits: int
    :type arms: int
    :type reward_means: numpy.ndarray
    """

    def __init__(
        self,
        bandits: int = 10,
        arms: int = 5,
        reward_means: np.ndarray = None,
        context_type: str = "tensor",
    ):
        super(GaussianMAB, self).__init__(bandits, arms, context_type)
        if reward_means is not None:
            self.reward_means = reward_means
        else:
            self.reward_means = np.random.random(size=(bandits, arms))

    def _compute_reward(self, action: int) -> Tuple[float, float]:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a gaussian distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken and maximum reward
        :rtype: tuples of int
        """
        reward_mean = self.reward_means[self.curr_bandit, action]
        reward = np.random.normal(reward_mean)
        return reward, max(self.reward_means[self.curr_bandit])
