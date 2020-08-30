from typing import Any, Dict

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from genrl.agents.deep.base import OnPolicyAgent
from genrl.utils import get_env_properties, get_model, safe_mean


class A2C(OnPolicyAgent):
    """Advantage Actor Critic algorithm (A2C)

    The synchronous version of A3C
    Paper: https://arxiv.org/abs/1602.01783

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
        rollout_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["rollout"]
        noise (:obj:`ActionNoise`): Action Noise function added to aid in exploration
        noise_std (float): Standard deviation of the action noise distribution
        value_coeff (float): Ratio of magnitude of value updates to policy updates
        entropy_coeff (float): Ratio of magnitude of entropy updates to policy updates
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        noise: Any = None,
        noise_std: float = 0.1,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        **kwargs
    ):
        super(A2C, self).__init__(*args, **kwargs)
        self.noise = noise
        self.noise_std = noise_std
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        """Function to initialize Actor-Critic architecture

        This will create the Actor-Critic net for the agent and initialise the action noise
        """
        state_dim, action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network
        )
        if isinstance(self.network, str):
            self.ac = get_model("ac", self.network)(
                state_dim,
                action_dim,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_type="V",
                discrete=discrete,
                action_lim=action_lim,
            ).to(self.device)
        else:
            self.ac = self.network.to(self.device)

            # action_dim = self.network.action_dim

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action given state

        Action Selection for On Policy Agents with Actor Critic

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
            value (:obj:`torch.Tensor`): Value of given state
            log_prob (:obj:`torch.Tensor`): Log probability of selected action
        """
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on actor output
        action, dist = self.ac.get_action(state, deterministic=deterministic)
        value = self.ac.get_value(state)

        return action.detach().cpu().numpy(), value, dist.log_prob(action).cpu()

    def get_traj_loss(self, values: torch.Tensor, dones: torch.Tensor) -> None:
        """Get loss from trajectory traversed by agent during rollouts

        Computes the returns and advantages needed for calculating loss

        Args:
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            dones (:obj:`list` of bool): Game over statuses of each environment
        """
        self.rollout.compute_returns_and_advantage(values.detach().cpu().numpy(), dones)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluates actions taken by actor

        Actions taken by actor and their respective states are analysed to get
        log probabilities and values from critics

        Args:
            states (:obj:`torch.Tensor`): States encountered in rollout
            actions (:obj:`torch.Tensor`): Actions taken in response to respective states

        Returns:
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            log_probs (:obj:`torch.Tensor`): Log of action probabilities given a state
        """
        states, actions = states.to(self.device), actions.to(self.device)
        _, dist = self.ac.get_action(states, deterministic=False)
        values = self.ac.get_value(states)
        return values, dist.log_prob(actions).cpu(), dist.entropy().cpu()

    def update_params(self) -> None:
        """Updates the the A2C network

        Function to update the A2C actor-critic architecture
        """
        for rollout in self.rollout.get(self.batch_size):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = self.evaluate_actions(
                rollout.observations, actions
            )

            policy_loss = rollout.advantages * log_prob
            policy_loss = -torch.mean(policy_loss)
            self.logs["policy_loss"].append(policy_loss.item())

            value_loss = self.value_coeff * F.mse_loss(rollout.returns, values.cpu())
            self.logs["value_loss"].append(torch.mean(value_loss).item())

            entropy_loss = -torch.mean(entropy)  # Change this to entropy
            self.logs["policy_entropy"].append(entropy_loss.item())

            actor_loss = policy_loss + self.entropy_coeff * entropy_loss

            self.optimizer_policy.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
            self.optimizer_value.step()

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
        """
        hyperparams = {
            "network": self.network,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "rollout_size": self.rollout_size,
            "weights": self.ac.state_dict(),
        }
        return hyperparams

    def load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`dict`): Dictionary of different neural net weights
        """
        self.ac.load_state_dict(weights["weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
            "policy_entropy": safe_mean(self.logs["policy_entropy"]),
            "mean_reward": safe_mean(self.rewards),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs"""
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["policy_entropy"] = []
        self.rewards = []
