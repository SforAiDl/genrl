from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from genrl.deep.agents import OffPolicyAgent
from genrl.deep.common.noise import ActionNoise
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean


class TD3:
    """Twin Delayed DDPG Algorithm

    Paper: https://arxiv.org/abs/1509.02971

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
        lr_value (float): Learning rate for the critic
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        polyak (float): Target model update parameter (1 for hard update)
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
        polyak: float = 0.995,
        policy_frequency: int = 2,
        noise: ActionNoise = None,
        noise_std: float = 0.1,
        **kwargs,
    ):
        super(TD3, self).__init(*args, **kwargs)
        self.polyak = polyak
        self.policy_frequency = policy_frequency
        self.noise = noise
        self.noise_std = noise_std

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, **kwargs) -> None:
        input_dim, action_dim, discrete, _ = get_env_properties(self.env, self.network)
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )

        if isinstance(self.network, str):
            self.ac = get_model("ac", self.network)(
                input_dim, action_dim, self.layers, "Qsa", False
            ).to(self.device)

            self.ac.qf2 = get_model("v", self.network)(
                state_dim, action_dim, hidden=self.layers, val_type="Qsa"
            )
        else:
            self.ac = self.network

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac.qf1 = self.ac.critic

        self.ac.qf1.to(self.device)
        self.ac.qf2.to(self.device)

        self.ac_target = deepcopy(self.ac).to(self.device)

        # freeze target network params
        for param in self.ac_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.replay_size, self.env)
        self.q_params = list(self.ac.qf1.parameters()) + list(self.ac.qf2.parameters())
        self.optimizer_q = torch.optim.Adam(self.q_params, lr=self.lr_q)

        self.optimizer_policy = torch.optim.Adam(
            self.ac.actor.parameters(), lr=self.lr_p
        )

    def update_params_before_select_action(self, timestep: int) -> None:
        """
        Update any parameters before selecting action like epsilon for decaying epsilon greedy

        :param timestep: Timestep in the training process
        :type timestep: int
        """
        pass

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action = self.ac_target.get_action(
                torch.as_tensor(state, dtype=torch.float32, device=self.device),
                deterministic=deterministic,
            )[0].numpy()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return np.clip(
            action, -self.env.action_space.high[0], self.env.action_space.high[0]
        )

    def get_q_loss(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> torch.Tensor:
        q1 = self.ac.qf1.get_value(torch.cat([state, action], dim=-1))
        q2 = self.ac.qf2.get_value(torch.cat([state, action], dim=-1))

        with torch.no_grad():
            target_q1 = self.ac_target.qf1.get_value(
                torch.cat(
                    [
                        next_state,
                        self.ac_target.get_action(next_state, deterministic=True)[0],
                    ],
                    dim=-1,
                )
            )
            target_q2 = self.ac_target.qf2.get_value(
                torch.cat(
                    [
                        next_state,
                        self.ac_target.get_action(next_state, deterministic=True)[0],
                    ],
                    dim=-1,
                )
            )
            target_q = torch.min(target_q1, target_q2).unsqueeze(1)

            target = reward.squeeze(1) + self.gamma * (1 - done) * target_q.squeeze(1)

        l1 = nn.MSELoss()(q1, target)
        l2 = nn.MSELoss()(q2, target)

        return l1 + l2

    def get_p_loss(self, state: np.array) -> torch.Tensor:
        q_pi = self.ac.get_value(
            torch.cat([state, self.ac.get_action(state, deterministic=True)[0]], dim=-1)
        )
        return -torch.mean(q_pi)

    def update_params(self, update_interval: int) -> None:
        for timestep in range(update_interval):
            batch = self.replay_buffer.sample(self.batch_size)
            state, action, reward, next_state, done = (x.to(self.device) for x in batch)
            self.optimizer_q.zero_grad()
            # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
            loss_q = self.get_q_loss(state, action, reward, next_state, done)
            loss_q.backward()
            self.optimizer_q.step()

            # Delayed Update
            if timestep % self.policy_frequency == 0:
                # freeze critic params for policy update
                for param in self.q_params:
                    param.requires_grad = False

                self.optimizer_policy.zero_grad()
                loss_p = self.get_p_loss(state)
                loss_p.backward()
                self.optimizer_policy.step()

                # unfreeze critic params
                for param in self.ac.critic.parameters():
                    param.requires_grad = True

                # update target network
                with torch.no_grad():
                    for param, param_target in zip(
                        self.ac.parameters(), self.ac_target.parameters()
                    ):
                        param_target.data.mul_(self.polyak)
                        param_target.data.add_((1 - self.polyak) * param.data)

                self.logs["policy_loss"].append(loss_p.item())
                self.logs["value_loss"].append(loss_q.item())

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "polyak": self.polyak,
            "policy_frequency": self.policy_frequency,
            "noise_std": self.noise_std,
            "q1_weights": self.ac.qf1.state_dict(),
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
