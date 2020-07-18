from typing import Tuple, Union

import gym
import torch

from ....environments import VecEnv
from .base import DQN


class DoubleDQN(DQN):
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
        super(DoubleDQN, self).__init__(
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

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        next_q_values = self.model(next_states)
        next_best_actions = next_q_values.max(1)[1].unsqueeze(1)

        next_q_target_values = self.target_model(next_states)
        max_next_q_target_values = next_q_target_values.gather(1, next_best_actions)
        target_q_values = rewards + self.gamma * max_next_q_target_values * (1 - dones)
        return target_q_values
