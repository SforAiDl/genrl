from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from torch import optim as opt
from torch.nn import functional as F

from genrl.deep.agents.base import BaseAgent
from genrl.deep.common import (
    PrioritizedBuffer,
    PushReplayBuffer,
    get_env_properties,
    get_model,
    safe_mean,
    set_seeds,
)
from genrl.environments import VecEnv


class BaseDQN(BaseAgent):
    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        batch_size: int = 32,
        gamma: float = 0.99,
        layers: Tuple = (32, 32),
        lr: float = 0.001,
        replay_size: int = 100,
        buffer_type: str = "push",
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 1000,
        **kwargs,
    ):
        super(BaseDQN, self).__init__(
            network_type,
            env,
            batch_size=batch_size,
            gamma=gamma,
            layers=layers,
            lr_policy=None,
            lr_value=lr,
            **kwargs,
        )
        self.replay_size = replay_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        if buffer_type == "push":
            self.buffer_class = PushReplayBuffer
        elif buffer_type == "prioritized":
            self.buffer_class = PrioritizedBuffer
        else:
            raise NotImplementedError

    def create_model(self, *args) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("v", self.network_type)(
            input_dim, action_dim, "Qs", self.layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def update_params_before_select_action(self, timestep: int) -> None:
        self.timestep = timestep
        self.epsilon = self.calculate_epsilon_by_frame()
        self.logs["epsilon"].append(self.epsilon)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:

        if not deterministic:
            if np.random.rand() < self.epsilon:
                return np.asarray(self.env.sample())

        state = torch.FloatTensor(state)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values, axis=-1)

    def get_q_values(self, states, actions) -> torch.Tensor:
        q_values = self.model(states).gather(1, actions)
        return q_values

    def get_target_q_values(self, next_states, rewards, dones):
        next_q_target_values = self.target_model(next_states)
        max_next_q_target_values = next_q_target_values.max(1)[0]

        target_q_values = rewards + self.gamma * torch.mul(
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values.unsqueeze(-1)

    def get_q_loss(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.reshape(-1, *self.env.obs_shape)
        actions = actions.reshape(-1, *self.env.action_shape).long()
        next_states = next_states.reshape(-1, *self.env.obs_shape)

        rewards = torch.FloatTensor(rewards).reshape(-1)
        dones = torch.FloatTensor(dones).reshape(-1)

        q_values = self.get_q_values(states, actions)
        target_q_values = self.get_target_q_values(next_states, rewards, dones)

        loss = F.mse_loss(q_values, target_q_values)
        self.logs["value_loss"].append(loss.item())
        return loss

    def update_params(self, update_interval: int) -> None:
        for timestep in range(update_interval):
            loss = self.get_q_loss()
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
            "lr": self.lr_value,
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
            "epsilon": safe_mean(self.logs["epsilon"]),
        }
        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["epsilon"] = []


class DQN(BaseDQN):
    def __init__(self, *args, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        self.empty_logs()
        self.create_model()
