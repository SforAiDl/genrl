import collections

import torch

from genrl.agents.deep.dqn.base import DQN


def ddqn_q_target(
    agent: DQN,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
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
    next_q_value_dist = agent.model(next_states)
    next_best_actions = torch.argmax(next_q_value_dist, dim=-1).unsqueeze(-1)

    rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

    next_q_target_value_dist = agent.target_model(next_states)
    max_next_q_target_values = next_q_target_value_dist.gather(2, next_best_actions)
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


def categorical_greedy_action(agent: DQN, state: torch.Tensor) -> torch.Tensor:
    """Greedy action selection for Categorical DQN

    Args:
        agent (:obj:`DQN`): The agent
        state (:obj:`torch.Tensor`): Current state of the environment

    Returns:
        action (:obj:`torch.Tensor`): Action taken by the agent
    """
    q_value_dist = agent.model(state.unsqueeze(0)).detach()  # .numpy()
    # We need to scale and discretise the Q-value distribution obtained above
    q_value_dist = q_value_dist * torch.linspace(
        agent.v_min, agent.v_max, agent.num_atoms
    )
    # Then we find the action with the highest Q-values for all discrete regions
    # Current shape of the q_value_dist is [1, n_envs, action_dim, num_atoms]
    # So we take the sum of all the individual atom q_values and then take argmax
    # along action dim to get the optimal action. Since batch_size is 1 for this
    # function, we squeeze the first dimension out.
    action = torch.argmax(q_value_dist.sum(-1), axis=-1).squeeze(0)
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
    q_value_dist = agent.model(states)

    # Size of q_value_dist should be [batch_size, n_envs, action_dim, num_atoms] here
    # To gather the q_values of the respective actions, actions must be of the shape:
    # [batch_size, n_envs, 1, num_atoms]. It's current shape is [batch_size, n_envs, 1]
    actions = actions.unsqueeze(-1).expand(
        agent.batch_size, agent.env.n_envs, 1, agent.num_atoms
    )
    # Now as we gather q_values from the action_dim dimension which is at index 2
    q_values = q_value_dist.gather(2, actions)

    # But after this the shape of q_values would be [batch_size, n_envs, 1, 51] where as
    # it needs to be the same as the target_q_values: [batch_size, n_envs, 51]
    q_values = q_values.squeeze(2)  # Hence the squeeze

    # Clamp Q-values to get positive and stable Q-values between 0 and 1
    q_values = q_values.clamp(0.01, 0.99)
    return q_values


def categorical_q_target(
    agent: DQN,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
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
    delta_z = float(agent.v_max - agent.v_min) / (agent.num_atoms - 1)
    support = torch.linspace(agent.v_min, agent.v_max, agent.num_atoms)

    next_q_value_dist = agent.target_model(next_states) * support
    next_actions = (
        torch.argmax(next_q_value_dist.sum(-1), axis=-1).unsqueeze(-1).unsqueeze(-1)
    )

    next_actions = next_actions.expand(
        agent.batch_size, agent.env.n_envs, 1, agent.num_atoms
    )
    next_q_values = next_q_value_dist.gather(2, next_actions).squeeze(2)

    rewards = rewards.unsqueeze(-1).expand_as(next_q_values)
    dones = dones.unsqueeze(-1).expand_as(next_q_values)

    # Refer to the paper in section 4 for notation
    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=agent.v_min, max=agent.v_max)
    bz = (Tz - agent.v_min) / delta_z
    l = bz.floor().long()
    u = bz.ceil().long()

    offset = (
        torch.linspace(
            0,
            (agent.batch_size * agent.env.n_envs - 1) * agent.num_atoms,
            agent.batch_size * agent.env.n_envs,
        )
        .long()
        .view(agent.batch_size, agent.env.n_envs, 1)
        .expand(agent.batch_size, agent.env.n_envs, agent.num_atoms)
    )

    target_q_values = torch.zeros(next_q_values.size())
    target_q_values.view(-1).index_add_(
        0,
        (l + offset).view(-1),
        (next_q_values * (u.float() - bz)).view(-1),
    )
    target_q_values.view(-1).index_add_(
        0,
        (u + offset).view(-1),
        (next_q_values * (bz - l.float())).view(-1),
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
    q_values = agent.get_q_values(batch.states, batch.actions)
    target_q_values = agent.get_target_q_values(
        batch.next_states, batch.rewards, batch.dones
    )

    # For the loss, we take the difference
    loss = -(target_q_values * q_values.log()).sum(1).mean()
    return loss
