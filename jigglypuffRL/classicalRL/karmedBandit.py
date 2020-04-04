import numpy as np
from abc import ABC


class Bandit(object):
    def __init__(self, bandits=1, arms=1):
        self.nbandits = bandits
        self.narms = arms
        self.bandits = np.random.normal((self.nbandits, self.narms))

    def learn(n_timesteps=None):
        raise NotImplementedError    
