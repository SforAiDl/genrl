import numpy as np
import torch.nn as nn

from jigglypuffRL.common.base import BaseValue
from jigglypuffRL.common.utils import mlp

def _get_val_model(arch, val_type, s_dim, hidden, a_dim=None,):
    if val_type is 'V':
        return arch([s_dim]+list(hidden)+[1])
    elif val_type is 'Qsa':
        return arch([s_dim+a_dim]+list(hidden)+[1])
    elif val_type is 'Qs':
        return arch([s_dim]+list(hidden)+[a_dim])


class MlpValue(BaseValue):
    def __init__(self, s_dim, a_dim=None, val_type='V', hidden=(32,32)):
        super(MlpValue, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.model = _get_val_model(mlp, val_type, s_dim, hidden, a_dim)