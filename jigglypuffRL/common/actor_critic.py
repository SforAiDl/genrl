import numpy as np
import torch.nn as nn

from jigglypuffRL.common.base import BaseActorCritic
from jigglypuffRL.common.policies import MlpPolicy
from jigglypuffRL.common.values import MlpValue

class MlpActorCritic(BaseActorCritic):
    def __init__(self, s_dim, a_dim, hidden=(32,32), val_type='V', disc=True, det=True, *args, **kwargs):
        super(MlpActorCritic, self).__init__(disc, det)

        self.actor = MlpPolicy(s_dim, a_dim, hidden, disc, det)
        self.critic = MlpValue(s_dim, a_dim, val_type, hidden)