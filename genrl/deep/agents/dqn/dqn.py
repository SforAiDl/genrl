from typing import Tuple, Union

import gym

from ....environments import VecEnv
from .base import BaseDQN


class DQN(BaseDQN):
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
        super(DQN, self).__init__(
            network_type,
            env,
            batch_size=batch_size,
            gamma=gamma,
            layers=layers,
            **kwargs,
        )
        self.replay_size = replay_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.layers = layers
        self.lr = lr

        self.empty_logs()
        self.create_model()
