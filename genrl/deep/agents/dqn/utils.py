import collections
from typing import List

import numpy as np
import torch
import torch.nn as nn

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


def prioritized_q_loss(agent: DQN, batch: collections.namedtuple):
    """Function to calculate the loss of the Q-function

    Returns:
        agent (:obj:`DQN`): The agent
        loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
    """
    batch = agent.sample_from_buffer(beta=agent.beta)

    q_values = agent.get_q_values(batch.states, batch.actions)
    target_q_values = agent.get_target_q_values(
        batch.next_states, batch.rewards, batch.dones
    )

    # Weighted MSE Loss
    loss = batch.weights * (q_values - target_q_values.detach()) ** 2
    # Priorities are taken as the td-errors + some small value to avoid 0s
    priorities = loss + 1e-5
    loss = loss.mean()
    agent.replay_buffer.update_priorities(
        batch.indices, priorities.detach().cpu().numpy()
    )
    agent.logs["value_loss"].append(loss.item())
    return loss


def categorical_greedy_action(agent: DQN, state: torch.Tensor) -> np.ndarray:
    """Greedy action selection for Categorical DQN

    Args:
        agent (:obj:`DQN`): The agent
        state (:obj:`np.ndarray`): Current state of the environment

    Returns:
        action (:obj:`np.ndarray`): Action taken by the agent
    """
    q_values = agent.model(state).detach().numpy()
    # We need to scale and discretise the Q-value distribution obtained above
    q_values = q_values * np.linspace(agent.v_min, agent.v_max, agent.num_atoms)
    # Then we find the action with the highest Q-values for all discrete regions
    action = np.argmax(q_values.sum(2), axis=-1)
    return action


def categorical_q_values(agent: DQN, states: torch.Tensor, actions: torch.Tensor):
    """Get Q values given state for a Categorical DQN

    Args:
        agent (:obj:`DQN`): The agent
        states (:obj:`torch.Tensor`): States being replayed
        actions (:obj:`torch.Tensor`): Actions being replayed

    Returns:
        q_values (:obj:`torch.Tensor`): Q values for the given states and actions
    """
    q_values = agent.model(states)

    # Size of q_values should be [..., action_dim, 51] here
    actions = actions.unsqueeze(1).expand(-1, 1, agent.num_atoms)
    q_values = q_values.gather(1, actions)

    # But after this the shape of q_values would be [..., 1, 51] where as
    # it needs to be the same as the target_q_values: [..., 51]
    q_values = q_values.squeeze(1)  # Hence the squeeze

    # Clamp Q-values to get positive and stable Q-values
    q_values = q_values.clamp(0.01, 0.99)

    return q_values


def categorical_q_target(
    agent: DQN, next_states: np.ndarray, rewards: List[float], dones: List[bool],
):
    """Projected Distribution of Q-values

    Helper function for Categorical/Distributional DQN

    Args:
        agent (:obj:`DQN`): The agent
        next_states (:obj:`torch.Tensor`): Next states being encountered by the agent
        rewards (:obj:`torch.Tensor`): Rewards received by the agent
        dones (:obj:`torch.Tensor`): Game over status of each environment

    Returns:
        target_q_values (object): Projected Q-value Distribution or Target Q Values
    """
    batch_size = next_states.size(0)

    delta_z = float(agent.v_max - agent.v_min) / (agent.num_atoms - 1)
    support = torch.linspace(agent.v_min, agent.v_max, agent.num_atoms)

    next_q_values = agent.target_model(next_states).data.cpu() * support
    next_action = torch.argmax(next_q_values.sum(2), axis=1)
    next_action = next_action[:, np.newaxis, np.newaxis].expand(
        next_q_values.size(0), 1, next_q_values.size(2)
    )
    next_q_values = next_q_values.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(-1).expand_as(next_q_values)
    dones = dones.unsqueeze(-1).expand_as(next_q_values)
    support = support.unsqueeze(0).expand_as(next_q_values)

    target_q_values = rewards + (1 - dones) * 0.99 * support
    target_q_values = target_q_values.clamp(min=agent.v_min, max=agent.v_max)
    norm_target_q_values = (target_q_values - agent.v_min) / delta_z
    lower = norm_target_q_values.floor()
    upper = norm_target_q_values.ceil()

    offset = (
        torch.linspace(0, (batch_size - 1) * agent.num_atoms, batch_size)
        .long()
        .unsqueeze(1)
        .expand(-1, agent.num_atoms)
    )

    target_q_values = torch.zeros(next_q_values.size())
    target_q_values.view(-1).index_add_(
        0,
        (lower.long() + offset).view(-1),
        (next_q_values * (upper - norm_target_q_values)).view(-1),
    )
    target_q_values.view(-1).index_add_(
        0,
        (upper.long() + offset).view(-1),
        (next_q_values * (norm_target_q_values - lower)).view(-1),
    )
    return target_q_values


def categorical_q_loss(agent: DQN, batch: collections.namedtuple):
    """Categorical DQN loss function to calculate the loss of the Q-function

    Args:
        agent (:obj:`DQN`): The agent
        batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

    Returns:
        loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
    """
    target_q_values = agent.get_target_q_values(
        batch.next_states, batch.rewards, batch.dones
    )
    q_values = agent.get_q_values(batch.states, batch.actions)

    # For the loss, we take the difference
    loss = -(target_q_values * q_values.log()).sum(1).mean()
    return loss
