import numpy as np


class Bandit(object):
    """
    Base Class for a Multi-armed Bandit
    :param arms: (int) Number of arms in the bandit
    """

    def __init__(self, arms=1):
        self._narms = arms

    @property
    def arms(self):
        return self._narms

    def step(self, action):
        raise NotImplementedError


class GaussianBandit(Bandit):
    """
    Bandit with Stationary Rewards following a Gaussian distribution.
    :param arms: (int) Number of arms in the bandit
    """

    def __init__(self, arms=1):
        super(GaussianBandit, self).__init__(arms)
        self._rewards = np.random.normal(size=arms)

    @property
    def rewards(self):
        return self._rewards

    def step(self, action):
        reward = np.random.normal(self.rewards[action])
        return reward


class BernoulliBandit(Bandit):
    """
    Multi-Armed Bandits with Bernoulli probabilities.
    :param arms: (int) Number of arms in the bandit
    :param reward
    """

    def __init__(self, arms=1, reward_probs=None):
        super(BernoulliBandit, self).__init__(arms)
        if reward_probs:
            self._rewards_probs = reward_probs
        else:
            self._rewards_probs = np.random.normal(size=(arms))

    @property
    def reward_probs(self):
        return self._rewards_probs

    def step(self, action):
        if np.random.random() < self._rewards_probs[action]:
            return 1
        else:
            return 0
