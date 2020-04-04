import numpy as np
from abc import ABC


class Bandit(object):
    def __init__(self, bandits=1, arms=1):
        self.nbandits = bandits
        self.narms = arms

    def learn(self, n_timesteps=None):
        raise NotImplementedError    


class BernoulliBandits(Bandit):
    def __init__(self, bandits=1, arms=1):
        super(BernoulliBandits, self).__init__(bandits, arms)
        self.bandits = np.random.random((self.nbandits, self.narms))

    def learn(self, n_timesteps=None):
        raise NotImplementedError


class GaussianBandits(Bandit):
    def __init(self, bandits=1, arms=1):
        super(GaussianBandits, self).__init__(bandits, arms)
        self.bandits = np.random.normal((self.nbandits, self.narms))

    def learn(self, n_timesteps=None):
        raise NotImplementedError
