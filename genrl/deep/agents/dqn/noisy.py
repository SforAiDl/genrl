from copy import deepcopy
from typing import Tuple, Union

import gym
import torch
from torch import optim as opt

from ....environments import VecEnv
from ...common import PrioritizedBuffer, PushReplayBuffer, get_env_properties, get_model
from .base import BaseDQN


class NoisyDQN(BaseDQN):
    def __init__(self, *args, noisy_layers: Tuple = (32, 128), **kwargs):
        super(NoisyDQN, self).__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers

        self.empty_logs()
        self.create_model()

    def create_model(self, *args) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "noisy")(
            input_dim, action_dim, fc_layers=self.layers, noisy_layers=self.noisy_layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)

    def update_params(self, update_interval: int) -> None:
        for timestep in range(update_interval):
            loss = self.get_q_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.reset_noise()
            self.target_model.reset_noise()

            if timestep % update_interval == 0:
                self.update_target_model()
