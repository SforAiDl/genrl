from typing import List

import numpy as np
import torch
from torch import nn as nn

from genrl.deep.agents.dqn.base import DQN


def ddqn_q_target(
    agent: DQN, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
) -> torch.Tensor:
    """Double Q-learning target

    Can be used to replace the `get_target_values` method of the Base DQN
    class in any DQN algorithm

    Args:
        agent (:obj:`DQN`): The agent
        next_states (:obj:`torch.Tensor`): Next states being encountered by the agent
        rewards (:obj:`torch.Tensor`): Rewards received by the agent
        dones (:obj:`torch.Tensor`): Game over status of each environment

    Returns:
        target_q_values (:obj:`torch.Tensor`): Target Q values using Double Q-learning
    """
    next_q_values = agent.model(next_states)
    next_best_actions = next_q_values.max(1)[1].unsqueeze(1)

    rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

    next_q_target_values = agent.target_model(next_states)
    max_next_q_target_values = next_q_target_values.gather(1, next_best_actions)
    target_q_values = rewards + agent.gamma * torch.mul(
        max_next_q_target_values, (1 - dones)
    )
    return target_q_values


def prioritized_q_loss(agent: DQN):
    """Function to calculate the loss of the Q-function

    Returns:
        loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
    """
    batch = agent.sample_from_buffer(beta=agent.beta)

    q_values = agent.get_q_values(batch.states, batch.actions)
    target_q_values = agent.get_target_q_values(
        batch.next_states, batch.rewards, batch.dones
    )

    loss = batch.weights * (q_values - target_q_values.detach()) ** 2
    priorities = loss + 1e-5
    loss = loss.mean()
    agent.replay_buffer.update_priorities(
        batch.indices, priorities.detach().cpu().numpy()
    )
    agent.logs["value_loss"].append(loss.item())
    return loss


def get_projection_distribution(
    agent: DQN, next_state: np.ndarray, rewards: List[float], dones: List[bool],
):
    """Projection Distribution

    Helper function for Categorical/Distributional DQN

    Args:
        next_states (:obj:`torch.Tensor`): Next states being encountered by the agent
        rewards (:obj:`torch.Tensor`): Rewards received by the agent
        dones (:obj:`torch.Tensor`): Game over status of each environment

    Returns:
        projection_distribution (object): Projection Distribution
    """
    batch_size = next_state.size(0)

    delta_z = float(agent.v_max - agent.v_min) / (agent.num_atoms - 1)
    support = torch.linspace(agent.v_min, agent.v_max, agent.num_atoms)

    next_dist = agent.target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = (
        next_action.unsqueeze(1)
        .unsqueeze(1)
        .expand(next_dist.size(0), 1, next_dist.size(2))
    )
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(-1)
    dones = dones.unsqueeze(-1)

    rewards = rewards.expand_as(next_dist)
    dones = dones.expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    tz = rewards + (1 - dones) * 0.99 * support
    tz = tz.clamp(min=agent.v_min, max=agent.v_max)
    bz = (tz - agent.v_min) / delta_z
    lower = bz.floor().long()
    upper = bz.ceil().long()

    offset = (
        torch.linspace(0, (batch_size - 1) * agent.num_atoms, batch_size)
        .long()
        .unsqueeze(1)
        .expand(-1, agent.num_atoms)
    )

    projection_distribution = torch.zeros(next_dist.size())
    projection_distribution.view(-1).index_add_(
        0, (lower + offset).view(-1), (next_dist * (upper.float() - bz)).view(-1)
    )
    projection_distribution.view(-1).index_add_(
        0, (upper + offset).view(-1), (next_dist * (bz - lower.float())).view(-1)
    )
    return projection_distribution
