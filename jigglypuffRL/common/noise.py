from abc import ABC, abstractmethod

import numpy as np


class ActionNoise(ABC):
    def __init__(self, mean, std):
        # super().__init__(mean, std)
        self._mean = mean
        self._std = std

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


class NormalActionNoise(ActionNoise):
    def __init__(self, mean, std):
        super(NormalActionNoise, self).__init__(mean, std)

    def __call__(self):
        return np.random.normal(self._mean, self._std)

    def reset(self):
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mean, std, theta=0.15, dt=1e-2, initial_noise=None):
        super(OrnsteinUhlenbeckActionNoise, self).__init__(mean, std)
        self._theta = theta
        self._mean = mean
        self._std = std
        self._dt = dt
        self._initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self):
        noise = (
            self.noise_prev
            + self._theta * (self._mean - self.noise_prev) * self._dt
            + self._std * np.sqrt(self._dt) * np.random.normal(size=self._mean.shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        self.noise_prev = (
            self._initial_noise
            if self._initial_noise is not None
            else np.zeros_like(self._mean)
        )
