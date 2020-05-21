import numpy as np


class Bandit(object):
    """
    Base Class for a Multi-armed Bandit

    :param arms: Number of arms in the bandit
    :type arms: int
    """

    def __init__(self, arms=1):
        self._narms = arms

    @property
    def arms(self):
        """
        Get the number of arms in the bandit

        :returns: Number of arms in the bandit
        :rtype: int
        """
        return self._narms

    def step(self, action):
        """
        Takes an action in the bandit and returns the sampled reward

        This method needs to be implemented in the specific bandit.

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int, float ...
        """
        raise NotImplementedError


class GaussianBandit(Bandit):
    """
    Bandit with Stationary Rewards following a Gaussian distribution.
    
    :param arms: Number of arms in the bandit
    :param rewards: The mean for the Gaussain distribution of each action
    :type arms: int
    :type rewards: numpy.ndarray
    """

    def __init__(self, arms=1, rewards=None):
        super(GaussianBandit, self).__init__(arms)
        if rewards:
            self._rewards = rewards
        else:
            self._rewards = np.random.normal(size=arms)

    @property
    def rewards(self):
        """
        Get the mean rewards for each action

        :returns: Mean reward for each action
        :rtype: numpy.ndarray
        """
        return self._rewards

    def step(self, action):
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a Gaussian distribution with a fixed mean

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: float
        """
        reward = np.random.normal(self.rewards[action])
        return reward


class BernoulliBandit(Bandit):
    """
    Multi-Armed Bandits with Bernoulli probabilities.

    :param arms: Number of arms in the bandit
    :param reward_probs: The probability of getting a reward for each action
    :type arms: int
    :type rewards: numpy.ndarray
    """

    def __init__(self, arms=1, reward_probs=None):
        super(BernoulliBandit, self).__init__(arms)
        if reward_probs:
            self._rewards_probs = reward_probs
        else:
            self._rewards_probs = np.random.normal(size=(arms))

    @property
    def reward_probs(self):
        """
        Get the probability of reward for each action

        :returns: Probability of reward for each action
        :rtype: numpy.ndarray
        """
        return self._rewards_probs

    def step(self, action):
        """
        Takes an action in the bandit and returns the sampled reward

        A reward of 1 is given with fixed probability.

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int
        """
        if np.random.random() < self._rewards_probs[action]:
            return 1
        else:
            return 0
