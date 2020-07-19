from copy import deepcopy
from typing import Tuple, Union

import gym
import torch
from torch import optim as opt

from ....environments import VecEnv
from ...common import PrioritizedBuffer, get_env_properties, get_model
from .base import BaseDQN


class PrioritizedReplayDQN(BaseDQN):
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
        alpha: float = 0.6,
        beta: float = 0.4,
        **kwargs,
    ):
        super(PrioritizedReplayDQN, self).__init__(
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
        self.alpha = alpha
        self.beta = beta

        self.empty_logs()
        self.create_model(PrioritizedBuffer, self.alpha)

    def get_q_loss(self) -> torch.Tensor:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay_buffer.sample(self.batch_size, self.beta)

        states = states.reshape(-1, *self.env.obs_shape)
        actions = actions.reshape(-1, *self.env.action_shape).long()
        next_states = next_states.reshape(-1, *self.env.obs_shape)

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.get_q_values(states, actions)
        target_q_values = self.get_target_q_values(next_states, rewards, dones)

        loss = weights * (q_values - target_q_values.detach()) ** 2
        priorities = loss + 1e-5
        loss = loss.mean()
        self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())
        self.logs["value_loss"].append(loss.item())
        return loss
