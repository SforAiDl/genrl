import torch
import torch.nn as nn


class DistributionalValue(torch.distributions.Categorical):
    def __init__(self, values: torch.Tensor, logits: torch.Tensor = None, probs: torch.Tensor = None):
        super(DistributionalValueDistribution, self).__init__():
            self._values = values

    @property
    def values(self):
        return self._values

class DiscreteValueDistributional(nn.Module):
    def __init__(self, prev_dim: int, num_atoms: int, V_min: float, V_max: float):
        super(DiscreteValueHead, self).__init__()
        self._linear = nn.Linear(prev_dim, num_atoms)
        self._values = torch.linspace(V_min, V_max, num_atoms)

    def forward(self, x: torch.Tensor) -> torch.distributions.distribution:
        logits = self._linear(x)
        values = self._values.type(logits.dtype)
        return DistributionalValueDistribution(values, logits)


class MultiVariateNormalDiagonalDistributional(nn.Module):
    def __init__(self, prev_dim: int, dimensions: int, act: torch.tanh = None, _init_scale: int = 0.3):
        super(MultiVariateNormalDiag, self).__init__()
        self._linear = nn.Linear(prev_dim, dimensions)
        self._act = act
        self._init_scale = _init_scale

    def forward(self, x: torch.Tensor):
        mean = self._linear(x)
        
        if act is not None:
            mean = self._act(mean)

        scale = torch.ones_like(mean) * self._init_scale
        return torch.distributions.MultiVariateNormalDiag(mean, scale)


class MixtureOfGaussians(nn.Module):
    def __init__(self, )