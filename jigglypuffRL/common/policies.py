import numpy as np
import torch.nn as nn

from base import BasePolicy

def mlp(sizes):
    layers = []
    for j in range(len(sizes) - 1):
        act = nn.ReLU if j < len(sizes) - 2 else nn.Identity
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MlpPolicy(BasePolicy):
    def __init__(self, s_dim, a_dim, hidden=(32,32), disc=True, det=True, **args, **kwargs):
        super(MlpPolicy, self).__init__(disc, det, **kwargs)

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.model = mlp([s_dim]+list(hidden)+[a_dim])