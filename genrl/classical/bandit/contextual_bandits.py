import numpy as np
from typing import Union


class ContextualBandit(object):
    """
    Base Class for a Multi-armed Bandit

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :type bandits: int
    :type arms: int
    """

    def __init__(self, bandits: int = 1, arms: int = 1):
        self._nbandits = bandits
        self._narms = arms

        self.reset()

    @property
    def arms(self) -> int:
        """
        Get the number of arms in each bandit

        :returns: Number of arms in each bandit
        :rtype: int
        """
        return self._narms

    @property
    def bandits(self) -> int:
        """
        Get the number of bandits

        :returns: Number of bandits
        :rtype: int
        """
        return self._nbandits

    def reset(self):
        """
        Resets the current bandit randomly
        
        :returns: The current bandit as observation
        :rtype: int
        """
        self.curr_bandit = np.random.randint(self.bandits)
        return self.curr_bandit

    def step(self, action: int) -> Union[int, float]:
        """
        Takes an action in the bandit and returns the sampled reward

        This method needs to be implemented in the specific bandit.

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int, float ...
        """
        raise NotImplementedError


class BernoulliCB(ContextualBandit):
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
        self, bandits: int = 1, arms: int = 1, reward_probs: np.ndarray = None
    ):
        super(BernoulliCB, self).__init__(bandits, arms)
        if reward_probs is not None:
            self.reward_probs = reward_probs
        else:
            self.reward_probs = np.random.random(size=(bandits, arms))

    def step(self, action: int) -> int:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a bernoulli distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int
        """
        reward_prob = self.reward_probs[self.curr_bandit, action]
        reward = int(np.random.random() > reward_prob)
        self.reset()
        return self.curr_bandit, reward


class GaussianCB(ContextualBandit):
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
        self, bandits: int = 1, arms: int = 1, reward_means: np.ndarray = None
    ):
        super(GaussianCB, self).__init__(bandits, arms)
        if reward_means is not None:
            self.reward_means = reward_means
        else:
            self.reward_means = np.random.random(size=(bandits, arms))

    def step(self, action: int) -> float:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a gaussian distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int
        """
        reward_mean = self.reward_means[self.curr_bandit, action]
        reward = np.random.normal(reward_mean)
        self.reset()
        return self.curr_bandit, reward
