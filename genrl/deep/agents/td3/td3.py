import collections
from copy import deepcopy
from typing import Any, Dict, List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from genrl.deep.agents import OffPolicyAgentAC
from genrl.deep.common.noise import ActionNoise
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean


class TD3(OffPolicyAgentAC):
    """Twin Delayed DDPG Algorithm

    Paper: https://arxiv.org/abs/1509.02971

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        policy_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the policy
        value_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the critics
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the critic
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        polyak (float): Target model update parameter (1 for hard update)
        policy_frequency (int): Frequency of policy updates in comparison to critic updates
        noise (:obj:`ActionNoise`): Action Noise function added to aid in exploration
        noise_std (float): Standard deviation of the action noise distribution
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        policy_frequency: int = 2,
        noise: ActionNoise = None,
        noise_std: float = 0.2,
        **kwargs,
    ):
        super(TD3, self).__init__(*args, **kwargs)
        self.policy_frequency = policy_frequency
        self.noise = noise
        self.noise_std = noise_std

        self.doublecritic = True

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        """Initializes class objects

        Initializes actor-critic architecture, replay buffer and optimizers
        """
        input_dim, action_dim, discrete, _ = get_env_properties(self.env, self.network)
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )

        if isinstance(self.network, str):
            # Below, the "12" corresponds to the Single Actor, Double Critic network architecture
            self.ac = get_model("ac", self.network + "12")(
                input_dim,
                action_dim,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_type="Qsa",
                discrete=False,
            )
        else:
            self.ac = self.network

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac_target = deepcopy(self.ac)

        self.replay_buffer = self.buffer_class(self.replay_size)
        self.critic_params = list(self.ac.critic1.parameters()) + list(
            self.ac.critic2.parameters()
        )
        self.optimizer_value = torch.optim.Adam(self.critic_params, lr=self.lr_value)
        self.optimizer_policy = torch.optim.Adam(
            self.ac.actor.parameters(), lr=self.lr_policy
        )

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()

            value_loss = self.get_q_loss(batch)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            # Delayed Update
            if timestep % self.policy_frequency == 0:
                policy_loss = self.get_p_loss(batch.states)

                for param in self.critic_params:
                    param.requires_grad = False

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                for param in self.critic_params:
                    param.requires_grad = True

                self.logs["policy_loss"].append(policy_loss.item())
                self.logs["value_loss"].append(value_loss.item())

                self.update_target_model()

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
        """
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_size": self.replay_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "polyak": self.polyak,
            "policy_frequency": self.policy_frequency,
            "noise_std": self.noise_std,
            "weights": self.ac.state_dict(),
        }

        return hyperparams

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs
        """
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
