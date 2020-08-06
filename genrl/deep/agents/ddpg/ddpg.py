import collections
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from genrl.deep.agents.base import OffPolicyAgent
from genrl.deep.common.noise import ActionNoise
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean, set_seeds
from genrl.environments import VecEnv


class DDPG(OffPolicyAgent):
    """DDPG Class

    Paper: https://arxiv.org/abs/1509.02971

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
        polyak (float): Soft target update coefficient
        noise (:obj:`ActionNoise`): Action Noise added to the policy output
        noise_std (float): Standard deviation of action noise
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        polyak: float = 0.995,
        noise: ActionNoise = None,
        noise_std: float = 0.1,
        **kwargs,
    ):
        super(DDPG, self).__init__(*args, **kwargs)
        self.polyak = polyak
        self.noise = noise
        self.noise_std = noise_std

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        input_dim, action_dim, discrete, _ = get_env_properties(
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
            input_dim, action_dim, self.layers, "Qsa", discrete=discrete
        ).to(self.device)

        self.ac_target = deepcopy(self.ac).to(self.device)

        self.replay_buffer = self.buffer_class(self.replay_size)
        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        for param, param_target in zip(
            self.ac.parameters(), self.ac_target.parameters()
        ):
            param_target.data.mul_(self.polyak)
            param_target.data.add_((1 - self.polyak) * param.data)

    def select_action(
        self, state: np.ndarray, deterministic: bool = True
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

    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q values corresponding to specific states and actions

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states

        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        q_values = self.ac.critic.get_value(torch.cat([states, actions], dim=-1))
        return q_values

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
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target_values
        return target_q_values

    def update_params(self, update_interval: int) -> None:
        """Takes the step for optimizer.

        Args:
            update_interval (int): No of timestep between target model updates
        """
        self.update_target_model()
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()

            value_loss = self.get_q_loss(batch)
            self.logs["value_loss"].append(value_loss.item())

            # freeze critic params for policy update
            for param in self.ac.critic.parameters():
                param.requires_grad = False

            policy_loss = self.get_p_loss(batch.states)
            self.logs["policy_loss"].append(policy_loss.item())

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # unfreeze critic params
            for param in self.ac.critic.parameters():
                param.requires_grad = True

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_size": self.replay_size,
            "polyak": self.polyak,
            "noise_std": self.noise_std,
            "lr_policy": self.lr_p,
            "lr_value": self.lr_q,
            "weights": self.ac.state_dict(),
        }
        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.load_state_dict(weights["weights"])

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
