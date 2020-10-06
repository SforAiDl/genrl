from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.optim as opt

from genrl.agents import OffPolicyAgentAC
from genrl.utils import get_env_properties, get_model, safe_mean


class SAC(OffPolicyAgentAC):
    """Soft Actor Critic algorithm (SAC)

    Paper: https://arxiv.org/abs/1812.05905

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        policy_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the policy
        value_layers (:obj:`tuple` of :obj:`int`): Neural network layer dimensions for the critics
        shared_layers(:obj:`tuple` of :obj:`int`): Sizes of shared layers in Actor Critic if using
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the critic
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        alpha (str): Entropy factor
        polyak (float): Target model update parameter (1 for hard update)
        entropy_tuning (bool): True if entropy tuning should be done, False otherwise
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
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

        self.doublecritic = True

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, **kwargs) -> None:
        """Initializes class objects

        Initializes actor-critic architecture, replay buffer and optimizers
        """
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

        if isinstance(self.network, str):
            state_dim, action_dim, discrete, _ = get_env_properties(
                self.env, self.network
            )
            arch = self.network + "12"
            if self.shared_layers is not None:
                arch += "s"
            self.ac = get_model("ac", arch)(
                state_dim,
                action_dim,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_type="Qsa",
                discrete=False,
                sac=True,
                action_scale=self.action_scale,
                action_bias=self.action_bias,
            )
        else:
            self.model = self.network

        self.ac_target = deepcopy(self.ac)
        actor_params, critic_params = self.ac.get_params()
        self.optimizer_value = opt.Adam(critic_params, self.lr_value)
        self.optimizer_policy = opt.Adam(actor_params, self.lr_policy)

        if self.entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.optimizer_alpha = opt.Adam([self.log_alpha], lr=self.lr_policy)

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action given state

        Action Selection

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        action, _, _ = self.ac.get_action(state, deterministic)
        return action.detach()

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        for param, param_target in zip(
            self.ac.parameters(), self.ac_target.parameters()
        ):
            param_target.data.mul_(self.polyak)
            param_target.data.add_((1 - self.polyak) * param.data)

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Get target Q values for the SAC

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the SAC
        """
        next_target_actions, next_log_probs, _ = self.ac.get_action(next_states)
        next_q_target_values = self.ac_target.get_value(
            torch.cat([next_states, next_target_actions], dim=-1), mode="min"
        ).squeeze() - self.alpha * next_log_probs.squeeze(1)
        target_q_values = rewards + self.gamma * (1 - dones) * next_q_target_values
        return target_q_values

    def get_p_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Function to get the Policy loss

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found

        Returns:
            loss (:obj:`torch.Tensor`): Calculated policy loss
        """
        next_best_actions, log_probs, _ = self.ac.get_action(states)
        q_values = self.ac.get_value(
            torch.cat([states, next_best_actions], dim=-1), mode="min"
        )
        policy_loss = ((self.alpha * log_probs) - q_values).mean()
        return policy_loss, log_probs

    def get_alpha_loss(self, log_probs):
        """Calculate Entropy Loss

        Args:
            log_probs (float): Log probs
        """
        if self.entropy_tuning:
            alpha_loss = -torch.mean(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            )
        else:
            alpha_loss = torch.FloatTensor(0.0)
        return alpha_loss

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            batch = self.sample_from_buffer()

            value_loss = self.get_q_loss(batch)
            self.logs["value_loss"].append(value_loss.item())

            policy_loss, log_probs = self.get_p_loss(batch.states)
            self.logs["policy_loss"].append(policy_loss.item())

            alpha_loss = self.get_alpha_loss(log_probs)
            self.logs["alpha_loss"].append(alpha_loss.item())

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()

            self.update_target_model()

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "lr_value": self.lr_value,
            "lr_policy": self.lr_policy,
            "replay_size": self.replay_size,
            "entropy_tuning": self.entropy_tuning,
            "alpha": self.alpha,
            "polyak": self.polyak,
        }
        return hyperparams, self.ac.state_dict()

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
            "alpha_loss": safe_mean(self.logs["alpha_loss"]),
        }
        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs"""
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["policy_loss"] = []
        self.logs["alpha_loss"] = []
