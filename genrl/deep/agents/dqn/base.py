from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import torch
from torch import optim as opt
from torch.nn import functional as F

from genrl.deep.agents.base import OffPolicyAgent
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean
from genrl.environments import VecEnv


class BaseDQN(OffPolicyAgent):
    """Base DQN class

    The Base class of any DQN algorithm. The methods can be changed as required.

    Attributes:
        network_type (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
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
        super(BaseDQN, self).__init__(*args, **kwargs)
        self.replay_size = replay_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def create_model(self, *args) -> None:
        """Function to initialize Q-value model

        This will create the Q-value function of the agent. Depends on the network type,
        types of Q-value models needed.

        Supported:
            "v" and "Qs" to `get_value_from_name` -> Regular Q-value function
            "dv" and "mlpdueling" -> Dueling Q-value function
            "dv" and "mlpnoisy" -> Noisy Q-value function
            "dv" and "mlpcategorical -> Categorical Q-value function

        You can replace "mlp" with "cnn" for using a CNN based Q-value function
        """
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("v", self.network_type)(
            input_dim, action_dim, "Qs", self.layers
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
        if not deterministic:
            if np.random.rand() < self.epsilon:
                return np.asarray(self.env.sample())

        state = torch.FloatTensor(state)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values, axis=-1)

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
        next_q_target_values = self.target_model(next_states)
        max_next_q_target_values = next_q_target_values.max(1)[0]

        target_q_values = rewards + self.gamma * torch.mul(
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values.unsqueeze(-1)

    def get_q_loss(self) -> torch.Tensor:
        """Function to calculate the loss of the Q-function

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.reshape(-1, *self.env.obs_shape)
        actions = actions.reshape(-1, *self.env.action_shape).long()
        rewards = torch.FloatTensor(rewards).reshape(-1)
        next_states = next_states.reshape(-1, *self.env.obs_shape)
        dones = torch.FloatTensor(dones).reshape(-1)

        q_values = self.get_q_values(states, actions)
        target_q_values = self.get_target_q_values(next_states, rewards, dones)
        loss = F.mse_loss(q_values, target_q_values)
        self.logs["value_loss"].append(loss.item())
        return loss

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            loss = self.get_q_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if timestep % update_interval == 0:
                self.update_target_model()

    def calculate_epsilon_by_frame(self) -> float:
        """Helper function to calculate epsilon after every timestep
        """
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


class DQN(BaseDQN):
    """Vanilla DQN class

    Simply calls the `create_model` function from the base class
    """

    def __init__(self, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        self.empty_logs()
        self.create_model()
