from copy import deepcopy
from typing import Tuple, Union

import gym
import torch
from torch import optim as opt

from ....environments import VecEnv
from ...common import PrioritizedBuffer, PushReplayBuffer, get_env_properties, get_model
from .base import BaseDQN


class DuelingDQN(BaseDQN):
    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        batch_size: int = 32,
        gamma: float = 0.99,
        layers: Tuple = (32, 32),
        lr: float = 0.001,
        replay_size: int = 100,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 1000,
        **kwargs,
    ):
        super(DuelingDQN, self).__init__(
            network_type,
            env,
            batch_size,
            gamma,
            layers,
            lr,
            replay_size,
            max_epsilon,
            min_epsilon,
            epsilon_decay,
            **kwargs,
        )
        self.empty_logs()
        self.create_model()

    def create_model(
        self,
        buffer_class: Union[PushReplayBuffer, PrioritizedBuffer] = PushReplayBuffer,
        *args,
    ) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "dueling")(
            input_dim, action_dim, self.layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)
