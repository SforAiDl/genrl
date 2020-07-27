import collections
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import torch
from torch import optim as opt
from torch.nn import functional as F

from genrl.deep.agents.base import OffPolicyAgent
from genrl.deep.common.buffers import (
    PrioritizedReplayBufferSamples,
    ReplayBufferSamples,
)
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean


class DQN(OffPolicyAgent):
    """Base DQN Class

    Paper: https://arxiv.org/abs/1312.5602

    Attributes:
        network_type (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_value (float): Learning rate for the Q-value function
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        max_epsilon (str): Maximum epsilon for exploration
        min_epsilon (str): Minimum epsilon for exploration
        epsilon_decay (str): Rate of decay of epsilon (in order to decrease
            exploration with time)
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 1000,
        **kwargs
    ):
        super(DQN, self).__init__(*args, **kwargs)
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.dqn_type = ""
        self.noisy = False

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, *args, **kwargs) -> None:
        """Function to initialize Q-value model

        This will create the Q-value function of the agent.
        """
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("v", self.network_type + self.dqn_type)(
            input_dim, action_dim, "Qs", self.layers, **kwargs
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def update_params_before_select_action(self, timestep: int) -> None:
        """Update necessary parameters before selecting an action

        This updates the epsilon (exploration rate) of the agent every timestep

        Args:
            timestep (int): Timestep of training
        """
        self.timestep = timestep
        self.epsilon = self.calculate_epsilon_by_frame()
        self.logs["epsilon"].append(self.epsilon)

    def get_greedy_action(self, state: torch.Tensor) -> np.ndarray:
        """Greedy action selection

        Args:
            state (:obj:`np.ndarray`): Current state of the environment

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        q_values = self.model(state).detach().numpy()
        action = np.argmax(q_values, axis=-1)
        return action

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action given state

        Epsilon-greedy action-selection

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        state = torch.FloatTensor(state)
        action = self.get_greedy_action(state)
        if not deterministic:
            if np.random.rand() < self.epsilon:
                action = np.asarray(self.env.sample())
        return action

    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q values corresponding to specific states and actions

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states

        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        q_values = self.model(states).gather(1, actions)
        return q_values

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Get target Q values for the DQN

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the DQN
        """
        next_q_target_values = self.target_model(
            next_states
        )  # Next Q-values according to target model
        max_next_q_target_values = next_q_target_values.max(1)[
            0
        ]  # Maximum of next q_target values
        target_q_values = rewards + self.gamma * torch.mul(  # Expected Target Q values
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values.unsqueeze(
            -1
        )  # Needs to be unsqueezed to match dimension of Q-values

    def sample_from_buffer(self, beta=None):
        """
        Samples experiences from the buffer and converts them into usable formats
        """
        # Samples from the buffer
        if beta is not None:
            batch = self.replay_buffer.sample(self.batch_size, beta=beta)
        else:
            batch = self.replay_buffer.sample(self.batch_size)

        # Parameters need to be reshaped and preprocessed before they're ready to send to Neural Networks.
        states = batch[0].reshape(-1, *self.env.obs_shape)
        actions = batch[1].reshape(-1, *self.env.action_shape).long()
        rewards = torch.FloatTensor(batch[2]).reshape(-1)
        next_states = batch[3].reshape(-1, *self.env.obs_shape)
        dones = torch.FloatTensor(batch[4]).reshape(-1)

        # Convert every experience to a Named Tuple. Either Replay or Prioritized Replay samples.
        if self.buffer_type == "push":
            batch = ReplayBufferSamples(*[states, actions, rewards, next_states, dones])
        elif self.buffer_type == "prioritized":
            indices, weights = batch[5], batch[6]
            batch = PrioritizedReplayBufferSamples(
                *[states, actions, rewards, next_states, dones, indices, weights]
            )
        return batch

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Normal Function to calculate the loss of the Q-function

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )
        loss = F.mse_loss(q_values, target_q_values)
        return loss

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()
            loss = self.get_q_loss(batch)
            self.logs["value_loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # In case the model uses Noisy layers, we must reset the noise every timestep
            if self.noisy:
                self.model.reset_noise()
                self.target_model.reset_noise()

            # Every few timesteps, we update the target Q network
            if timestep % update_interval == 0:
                self.update_target_model()

    def calculate_epsilon_by_frame(self) -> float:
        """Helper function to calculate epsilon after every timestep
        """
        # Exponentially decays exploration rate from max epsilon to min epsilon
        # The greater the value of epsilon_decay, the slower the decrease in epsilon
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -1.0 * self.timestep / self.epsilon_decay
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
        """
        hyperparams = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr": self.lr_value,
            "replay_size": self.replay_size,
            "weights": self.model.state_dict(),
            "timestep": self.timestep,
        }
        return hyperparams

    def load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model
        """
        self.model.load_state_dict(weights["weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "value_loss": safe_mean(self.logs["value_loss"]),
            "epsilon": safe_mean(self.logs["epsilon"]),
        }
        self.empty_logs()
        return logs

    def empty_logs(self) -> None:
        """Empties logs
        """
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["epsilon"] = []
