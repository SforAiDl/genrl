from typing import Tuple

import numpy as np

from genrl.core.bandit import MultiArmedBandit


class BernoulliMAB(MultiArmedBandit):
    """
    Contextual Bandit with categorial context and bernoulli reward distribution

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param reward_probs: Probabilities of getting rewards
    :type bandits: int
    :type arms: int
    :type reward_probs: numpy.ndarray
    """

    def __init__(
        self,
        bandits: int = 1,
        arms: int = 5,
        reward_probs: np.ndarray = None,
        context_type: str = "tensor",
    ):
        super(BernoulliMAB, self).__init__(bandits, arms, context_type)
        if reward_probs is not None:
            self.reward_probs = reward_probs
        else:
            self.reward_probs = np.random.random(size=(bandits, arms))

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a bernoulli distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken and maximum reward
        :rtype: tuple of int
        """
        reward_prob = self.reward_probs[self.curr_bandit, action]
        reward = int(np.random.random() > reward_prob)
        return reward, 1
