import collections
from typing import List

import torch
from torch.nn import functional as F

from genrl.agents.deep.base import BaseAgent
from genrl.core import (
    PrioritizedBuffer,
    PrioritizedReplayBufferSamples,
    ReplayBuffer,
    ReplayBufferSamples,
)


class OffPolicyAgent(BaseAgent):
    """Off Policy Agent Base Class

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self, *args, replay_size: int = 5000, buffer_type: str = "push", **kwargs
    ):
        super(OffPolicyAgent, self).__init__(*args, **kwargs)
        self.replay_size = replay_size

        if buffer_type == "push":
            self.replay_buffer = ReplayBuffer(self.replay_size)
        elif buffer_type == "prioritized":
            self.replay_buffer = PrioritizedBuffer(self.replay_size)
        else:
            raise NotImplementedError

    def update_params_before_select_action(self, timestep: int) -> None:
        """Update any parameters before selecting action like epsilon for decaying epsilon greedy

        Args:
            timestep (int): Timestep in the training process
        """
        pass

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model"""
        raise NotImplementedError

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        raise NotImplementedError

    def _reshape_batch(self, batch: List):
        """Function to reshape experiences

        Can be modified for individual algorithm usage

        Args:
            batch (:obj:`list`): List of experiences that are being replayed

        Returns:
            batch (:obj:`list`): Reshaped experiences for replay
        """
        return [*batch]

    def sample_from_buffer(self, beta: float = None):
        """Samples experiences from the buffer and converts them into usable formats

        Args:
            beta (float): Importance-Sampling beta for prioritized replay

        Returns:
            batch (:obj:`list`): Replay experiences sampled from the buffer
        """
        # Samples from the buffer
        if beta is not None:
            batch = self.replay_buffer.sample(self.batch_size, beta=beta)
        else:
            batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = self._reshape_batch(batch)

        # Convert every experience to a Named Tuple. Either Replay or Prioritized Replay samples.
        if isinstance(self.replay_buffer, ReplayBuffer):
            batch = ReplayBufferSamples(*[states, actions, rewards, next_states, dones])
        elif isinstance(self.replay_buffer, PrioritizedBuffer):
            indices, weights = batch[5], batch[6]
            batch = PrioritizedReplayBufferSamples(
                *[states, actions, rewards, next_states, dones, indices, weights]
            )
        else:
            raise NotImplementedError
        return batch

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Normal Function to calculate the loss of the Q-function or critic

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )
        loss = F.mse_loss(q_values, target_q_values)
        return loss


class OffPolicyAgentAC(OffPolicyAgent):
    """Off Policy Agent Base Class

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, polyak=0.995, **kwargs):
        super(OffPolicyAgentAC, self).__init__(*args, **kwargs)
        self.polyak = polyak
        self.doublecritic = False

    def select_action(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Select action given state

        Deterministic Action Selection with Noise

        Args:
            state (:obj:`torch.Tensor`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        action, _ = self.ac.get_action(state, deterministic)
        action = action.detach()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return torch.clamp(
            action, self.env.action_space.low[0], self.env.action_space.high[0]
        )

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        for param, param_target in zip(
            self.ac.parameters(), self.ac_target.parameters()
        ):
            param_target.data.mul_(self.polyak)
            param_target.data.add_((1 - self.polyak) * param.data)

    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q values corresponding to specific states and actions

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states

        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        if self.doublecritic:
            q_values = self.ac.get_value(
                torch.cat([states, actions], dim=-1), mode="both"
            )
        else:
            q_values = self.ac.get_value(torch.cat([states, actions], dim=-1))
        return q_values

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Get target Q values for the TD3

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the TD3
        """
        next_target_actions = self.ac_target.get_action(next_states, True)[0]

        if self.doublecritic:
            next_q_target_values = self.ac_target.get_value(
                torch.cat([next_states, next_target_actions], dim=-1), mode="min"
            )
        else:
            next_q_target_values = self.ac_target.get_value(
                torch.cat([next_states, next_target_actions], dim=-1)
            )
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target_values

        return target_q_values

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Actor Critic Function to calculate the loss of the Q-function or critic

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )
        if self.doublecritic:
            loss = F.mse_loss(q_values[0], target_q_values) + F.mse_loss(
                q_values[1], target_q_values
            )
        else:
            loss = F.mse_loss(q_values, target_q_values)
        return loss

    def get_p_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Function to get the Policy loss

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found

        Returns:
            loss (:obj:`torch.Tensor`): Calculated policy loss
        """
        next_best_actions = self.ac.get_action(states, True)[0]
        q_values = self.ac.get_value(torch.cat([states, next_best_actions], dim=-1))
        policy_loss = -torch.mean(q_values)
        return policy_loss

    def _load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`torch.Tensor`): neural net weights
        """
        self.ac.load_state_dict(weights)
