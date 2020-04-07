import numpy as np
import torch.nn as nn

from jigglypuffRL.common.base import BasePolicy
from jigglypuffRL.common.utils import mlp

class MlpPolicy(BasePolicy):
    """
    MLP Policy
    :param s_dim: (int) state dimension of environment
    :param a_dim: (int) action dimension of environment
    :param hidden: (tuple or list) sizes of hidden layers
    :param disc: (bool) discrete action space?
    :param det: (bool) deterministic policy?
    """
    def __init__(self, s_dim, a_dim, hidden=(32,32), disc=True, det=True, *args, **kwargs):
        super(MlpPolicy, self).__init__(disc, det, **kwargs)

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.model = mlp([s_dim]+list(hidden)+[a_dim])