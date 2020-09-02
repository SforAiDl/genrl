import math
from abc import ABC, abstractmethod

import torch  # noqa
import torch.nn as nn  # noqa


class ActionNoise(ABC):
    """
    Base class for Action Noise

    :param mean: Mean of noise distribution
    :param std: Standard deviation of noise distribution
    :type mean: float
    :type std: float
    """

    def __init__(self, mean: float, std: float):
        # super().__init__(mean, std)
        self._mean = mean
        self._std = std

    @abstractmethod
    def __call__(self) -> None:
        raise NotImplementedError

    @property
    def mean(self) -> float:
        """
        Returns mean of noise distribution
        """
        return self._mean

    @property
    def std(self) -> float:
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

    def __init__(self, mean: float, std: float):
        super(NormalActionNoise, self).__init__(mean, std)

    def __call__(self) -> float:
        """
        Return action noise randomly sampled from noise distribution
        """
        return torch.normal(self._mean, self._std)

    def reset(self) -> None:
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
    :type initial_noise: torch.Tensor
    """

    def __init__(
        self,
        mean: float,
        std: float,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: torch.Tensor = None,
    ):
        super(OrnsteinUhlenbeckActionNoise, self).__init__(mean, std)
        self._theta = theta
        self._mean = mean
        self._std = std
        self._dt = dt
        self._initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self) -> float:
        """
                (Return action noise randomly sampled from noise distribution
        according to the Ornstein Uhlenbeck process)
        """
        noise = (
            self.noise_prev
            + self._theta * (self._mean - self.noise_prev) * self._dt
            + (self._std * math.sqrt(self._dt) * torch.randn(self._mean.shape))
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        Reset the initial noise value for the noise distribution sampling
        """
        self.noise_prev = (
            self._initial_noise
            if self._initial_noise is not None
            else torch.zeros(self._mean.shape)
        )


class NoisyLinear(nn.Module):
    """Noisy Linear Layer Class

    Class to represent a Noisy Linear class (noisy version of nn.Linear)

    Attributes:
        in_features (int): Input dimensions
        out_features (int): Output dimensions
        std_init (float): Weight initialisation constant
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(state, weight, bias)

    def reset_parameters(self) -> None:
        """Reset parameters of layer"""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self) -> None:
        """Reset noise components of layer"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        inp = torch.randn(size)
        inp = inp.sign().mul(inp.abs().sqrt())
        return inp
