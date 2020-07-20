from copy import deepcopy

import torch
from torch import optim as opt

from genrl.deep.agents.dqn.base import BaseDQN


class PrioritizedReplayDQN(BaseDQN):
    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super(PrioritizedReplayDQN, self).__init__(
            *args, buffer_type="prioritized", **kwargs
        )
        self.alpha = alpha
        self.beta = beta

        self.empty_logs()
        self.create_model(self.alpha)

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

        rewards = torch.FloatTensor(rewards).reshape(-1)
        dones = torch.FloatTensor(dones).reshape(-1)

        q_values = self.get_q_values(states, actions)
        target_q_values = self.get_target_q_values(next_states, rewards, dones)

        loss = weights * (q_values - target_q_values.detach()) ** 2
        priorities = loss + 1e-5
        loss = loss.mean()
        self.replay_buffer.update_priorities(indices, priorities.detach().cpu().numpy())
        self.logs["value_loss"].append(loss.item())
        return loss
