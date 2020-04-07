import numpy as np
import torch.nn as nn

from base import BasePolicy
from utils import mlp

class MlpPolicy(BasePolicy):
    def __init__(self, s_dim, a_dim, hidden=(32,32), disc=True, det=True, **args, **kwargs):
        super(MlpPolicy, self).__init__(disc, det, **kwargs)

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.model = mlp([s_dim]+list(hidden)+[a_dim])