from abc import ABC, abstractmethod

import numpy as np


class ActionNoise(ABC):
    """
    Base class for Action Noise

    :param mean: Mean of noise distribution
    :param std: Standard deviation of noise distribution
    :type mean: float
    :type std: float
    """
    def __init__(self, mean, std):
        # super().__init__(mean, std)
        self._mean = mean
        self._std = std

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @property
    def mean(self):
        """
        Returns mean of noise distribution
        """
        return self._mean

    @property
    def std(self):
        """
        Returns standard deviation of noise distribution
        """
        return self._std


class NormalActionNoise(ActionNoise):
    """
    Normal implementation of Action Noise

    :param mean: Mean of noise distribution
    :param std: Standard deviation of noise distribution
    :type mean: float
    :type std: float
    """
    def __init__(self, mean, std):
        super(NormalActionNoise, self).__init__(mean, std)

    def __call__(self):
        """
        Return action noise randomly sampled from noise distribution
        """
        return np.random.normal(self._mean, self._std)

    def reset(self):
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    Ornstein Uhlenbeck implementation of Action Noise

    :param mean: Mean of noise distribution
    :param std: Standard deviation of noise distribution
    :param theta: Parameter used to solve the Ornstein Uhlenbeck process
    :param dt: Small parameter used to solve the Ornstein Uhlenbeck process
    :param initial_noise: Initial noise distribution
    :type mean: float
    :type std: float
    :type theta: float
    :type dt: float
    :type initial_noise: Numpy array
    """
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
        """
        Return action noise randomly sampled from noise distribution \
according to the Ornstein Uhlenbeck process
        """
        noise = (
            self.noise_prev
            + self._theta * (self._mean - self.noise_prev) * self._dt
            + (
                self._std
                * np.sqrt(self._dt)
                * np.random.normal(size=self._mean.shape)
            )
        )
        self.noise_prev = noise
        return noise

    def reset(self):
        """
        Reset the initial noise value for the noise distribution sampling
        """
        self.noise_prev = (
            self._initial_noise
            if self._initial_noise is not None
            else np.zeros_like(self._mean)
        )
