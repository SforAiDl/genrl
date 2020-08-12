from copy import deepcopy
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.distributions import Normal

from genrl.deep.agents.base import OffPolicyAgent
from genrl.deep.common.base import BaseActorCritic
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean


class SAC(OffPolicyAgent):
    """Soft Actor Critic algorithm (SAC)

    Paper: https://arxiv.org/abs/1812.05905
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.01,
        polyak: float = 0.995,
        entropy_tuning: bool = True,
        **kwargs,
    ):
        super(SAC, self).__init__(*args, **kwargs)

        self.alpha = alpha
        self.polyak = polyak
        self.entropy_tuning = entropy_tuning

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, **kwargs) -> None:
        if isinstance(self.network, str):
            input_dim, action_dim, discrete, _ = get_env_properties(
                self.env, self.network
            )

            self.ac = get_model("ac", self.network + "12")(
                input_dim,
                action_dim,
                hidden=self.layers,
                val_type="Qsa",
                discrete=False,
                num_critics=2,
                sac=True,
            ).float()
        else:
            self.model = self.network

        self.ac_target = deepcopy(self.ac)

        self.critic_params = list(self.ac.critic[0].parameters()) + list(
            self.ac.critic[1].parameters()
        )

        self.optimizer_value = opt.Adam(self.critic_params, self.lr_value)
        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), self.lr_policy)

        if self.entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.optimizer_alpha = opt.Adam([self.log_alpha], lr=self.lr_policy)

        self.replay_buffer = self.buffer_class(self.replay_size)

        # set action scales
        if self.env.action_space is None:
            self.action_scale = torch.FloatTensor(1.0)
            self.action_bias = torch.FloatTensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (self.env.action_space.high + self.env.action_space.low) / 2.0
            )

    def sample_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        mean, log_std = self.ac.actor.forward(state)
        std = log_std.exp()

        # reparameterization trick
        distribution = Normal(mean, std)
        xi = distribution.rsample()
        yi = torch.tanh(xi)
        action = yi * self.action_scale + self.action_bias
        log_pi = distribution.log_prob(xi)

        # enforcing action bound (appendix of paper)
        log_pi -= torch.log(
            self.action_scale * (1 - yi.pow(2)) + np.finfo(np.float32).eps
        )
        log_pi = log_pi.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.float(), log_pi, mean

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.sample_action(state, deterministic)
        return action.detach().cpu().numpy()

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
        q_values = self.ac.get_value(torch.cat([states, actions], dim=-1), mode="both")
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
        next_target_actions, next_log_pi, _ = self.sample_action(next_states)
        next_q_target_values = self.ac_target.get_value(
            torch.cat([next_states, next_target_actions], dim=-1), mode="min"
        ) - self.alpha * next_log_pi.squeeze(-1)
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target_values
        return target_q_values

    def get_q_loss(self, batch: NamedTuple) -> torch.Tensor:
        """TD3 Function to calculate the loss of the critic

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )
        loss = F.mse_loss(q_values[0], target_q_values) + F.mse_loss(
            q_values[1], target_q_values
        )
        return loss

    def get_p_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Function to get the Policy loss

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found

        Returns:
            loss (:obj:`torch.Tensor`): Calculated policy loss
        """
        pi, log_pi, _ = self.sample_action(states)
        critic_pi = self.ac.get_value(
            torch.cat([states, pi.float()], dim=-1).float(), mode="min"
        )
        policy_loss = ((self.alpha * log_pi) - critic_pi).mean()

        return policy_loss, log_pi

    def get_alpha_loss(self, log_pi):
        # Entropy loss
        if self.entropy_tuning:
            alpha_loss = -torch.mean(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            )
        else:
            alpha_loss = torch.FloatTensor(0.0)
            self.alpha = self.log_alpha.exp()
        return alpha_loss

    def update_params(self, update_interval: int) -> (Tuple[float]):
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()

            value_loss = self.get_q_loss(batch)
            self.logs["value_loss"].append(value_loss.item())

            policy_loss, log_pi = self.get_p_loss(batch.states)
            self.logs["policy_loss"].append(policy_loss.item())

            alpha_loss = self.get_alpha_loss(log_pi)
            self.logs["alpha_loss"].append(alpha_loss.item())

            policy_loss += alpha_loss

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.update_target_model()

        # self.logs["value_loss"].append(value_loss.item())
        # self.logs["policy_loss"].append(policy_loss.item())
        # self.logs["alpha_loss"].append(alpha_loss.item())

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "lr_value": self.lr_value,
            "lr_policy": self.lr_policy,
            "replay_size": self.replay_size,
            "entropy_tuning": self.entropy_tuning,
            "alpha": self.alpha,
            "polyak": self.polyak,
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
            "alpha_loss": safe_mean(self.logs["alpha_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["policy_loss"] = []
        self.logs["alpha_loss"] = []
