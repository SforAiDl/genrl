from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch
from torch import optim as opt
from torch.nn import functional as F

from ....environments import VecEnv
from ...common import get_env_properties, get_model, safe_mean, set_seeds


class DQN(BaseAgent):
    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        gamma: float = 0.99,
        lr: float = 0.001,
        batch_size: int = 32,
        replay_size: int = 100,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 1000,
        layers: Tuple = (32, 32) ** kwargs,
    ):
        self.network_type = network_type
        self.env = env
        self.replay_size = replay_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.layers = layers

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # Setup tensorboard writer
        self.writer = None

        self.empty_logs()
        self.create_model()

    def create_model(self) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("v", self.network_type)(
            input_dim, action_dim, "Qs", self.layers
        )

        self.target_model = deepcopy(self.model)

        self.replay_buffer = ReplayBuffer(self.replay_size, self.env)

        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def update_params_before_select_action(self, timestep: int) -> None:
        self.timestep = timestep
        self.epsilon = self.calculate_epsilon_by_frame()

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:

        if not deterministic:
            if np.random.rand() < self.epsilon:
                return np.asarray(self.env.sample())

        state = Variable(torch.FloatTensor(state))
        q_values = self.model(state).detach().numpy()

        return np.argmax(q_values, axis=-1)

    def get_q_values(self) -> torch.Tensor:
        q_values = self.model(states).gather(1, actions)
        return q_values

    def get_target_q_values(self):
        next_q_values = self.target_model(next_states)
        max_next_q_values = q_next_state_values.max(1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        return target_q_values

    def update_params(self, update_interval: int) -> None:
        for timestep in range(update_interval):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

            states = states.reshape(-1, *self.env.obs_shape)
            actions = actions.reshape(-1, *self.env.action_shape).long()
            next_states = next_states.reshape(-1, *self.env.obs_shape)

            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            q_values = self.get_q_values(states, actions)
            target_q_values = self.get_target_q_values(next_states, rewards, dones)

            loss = F.mse_loss(q_values, target_q_values)
            self.logs["value_loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if timestep % update_interval == 0:
                self.update_target_model()

    def calculate_epsilon_by_frame(self) -> float:
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -1.0 * self.timestep / self.epsilon_decay
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "replay_size": self.replay_size,
            "weights": self.model.state_dict(),
            "timestep": self.timestep,
        }
        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.model.load_state_dict(weights["weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """
        logs = {
            "value_loss": safe_mean(self.logs["value_loss"]),
        }
        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["value_loss"] = []
