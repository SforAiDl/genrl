from copy import deepcopy
from typing import Tuple, Union

import gym
import torch
from torch import optim as opt

from ....environments import VecEnv
from ...common import PrioritizedBuffer, PushReplayBuffer, get_env_properties, get_model
from .base import BaseDQN


class DuelingDQN(BaseDQN):
    def __init__(self, *args, **kwargs):
        super(DuelingDQN, self).__init__(*args, **kwargs)
        self.empty_logs()
        self.create_model()

    def create_model(self, *args) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "dueling")(
            input_dim, action_dim, self.layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)
