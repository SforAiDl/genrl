import collections
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from genrl.deep.agents import OffPolicyAgent
from genrl.deep.common import (
    ReplayBuffer,
    get_env_properties,
    get_model,
    safe_mean,
    set_seeds,
)
from genrl.environments import VecEnv


class TD3(OffPolicyAgent):
    """
    Twin Delayed DDPG

    Paper: https://arxiv.org/abs/1509.02971
    """

    def __init__(
        self,
        *args,
        polyak: float = 0.995,
        noise: Optional[Any] = None,
        noise_std: float = 0.1,
        **kwargs,
    ):
        super(TD3, self).__init__(*args, **kwargs)
        self.polyak = polyak
        self.noise = noise
        self.noise_std = noise_std

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        state_dim, action_dim, discrete, _ = get_env_properties(
            self.env, self.network_type
        )
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )
        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "Qsa", False
        )
        self.ac_target = deepcopy(self.ac).to(self.device)

        self.replay_buffer = self.buffer_class(self.env)
        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state = torch.as_tensor(state).float()
        action, _ = self.ac.get_action(state, deterministic)
        action = action.detach().cpu().numpy()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return np.clip(
            action, self.env.action_space.low[0], self.env.action_space.high[0]
        )

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Get target Q values for the DDPG

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the DQN
        """
        next_target_actions = self.ac_target.get_action(next_states, True)[0]
        next_q_target_values = self.ac_target.get_value(
            torch.cat([next_states, next_target_actions], dim=-1)
        )
        min_next_q_target_values = torch.min(*next_q_target_values)
        target_q_values = rewards + self.gamma * (1 - dones) * min_next_q_target_values
        return target_q_values

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Normal Function to calculate the loss of the Q-function

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )

        return sum(
            [F.mse_loss(q_values[i], target_q_values) for i in range(len(q_values))]
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

    def update_params(self, update_interval: int) -> None:
        """Takes the step for optimizer.

        Args:
            update_interval (int): No of timestep between target model updates
        """
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()
            value_loss = self.get_q_loss(batch)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            if timestep % self.policy_frequency == 0:
                # freeze critic params for policy update
                for param in self.q_params:
                    param.requires_grad = False

                policy_loss = self.get_p_loss(batch.states)
                self.logs["policy_loss"].append(policy_loss.item())
                self.logs["value_loss"].append(value_loss.item())

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                # unfreeze critic params
                for param in self.ac.critic.parameters:
                    param.requires_grad = True

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "gamma": self.gamma,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "polyak": self.polyak,
            "policy_frequency": self.policy_frequency,
            "noise_std": self.noise_std,
            "weights": self.ac.state_dict(),
            "q2_weights": self.ac.qf2.state_dict(),
            "actor_weights": self.ac.actor.state_dict(),
        }

        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.actor.load_state_dict(weights["actor_weights"])
        self.ac.qf1.load_state_dict(weights["q1_weights"])
        self.ac.qf2.load_state_dict(weights["q2_weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
