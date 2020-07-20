from typing import Tuple, Union

import gym
import torch

from genrl.deep.agents.dqn.base import BaseDQN


class DoubleDQN(BaseDQN):
    def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, **kwargs)
        self.empty_logs()
        self.create_model()

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        next_q_values = self.model(next_states)
        next_best_actions = next_q_values.max(1)[1].unsqueeze(1)

        rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

        next_q_target_values = self.target_model(next_states)
        max_next_q_target_values = next_q_target_values.gather(1, next_best_actions)
        target_q_values = rewards + self.gamma * torch.mul(
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values
