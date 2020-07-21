from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from ....environments.vec_env import VecEnv
from ...common import RolloutBuffer, get_env_properties, get_model, safe_mean
from ..base import OnPolicyAgent


class A2C(OnPolicyAgent):
    """
    Advantage Actor Critic algorithm (A2C)
    The synchronous version of A3C
    Paper: https://arxiv.org/abs/1602.01783

    Attributes:
        network_type (str): model architecture type ['mlp','cnn']
        env (gym.Env or VectorEnv): environment for agent to interact with
        batch_size (int): number of transitions per batch
        gamma (float): discount factor
        lr_policy (float): policy optimizer learning rate
        lr_value (float): value function optimizer learning rate
        epochs (int): number of training episodes
        max_ep_len (int): maximum episode length
        layers (tuple(int)): number of hidden units in layers of network
        noise (Any): noise type
        noise_std (float): standard deviation for action noise
        rollout_size (int): size of rollout buffer
        value_coeff (float): coefficient of value function loss in loss function
        entropy_coeff (float): coefficient of entropy loss in loss function
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        batch_size: int = 256,
        gamma: float = 0.99,
        lr_policy: float = 0.01,
        lr_value: float = 0.1,
        epochs: int = 100,
        max_ep_len: int = 1000,
        layers: Tuple = (32, 32),
        noise: Any = None,
        noise_std: float = 0.1,
        rollout_size: int = 2048,
        **kwargs
    ):

        super(A2C, self).__init__(
            network_type,
            env,
            batch_size,
            layers,
            gamma,
            lr_policy,
            lr_value,
            epochs,
            rollout_size,
            **kwargs
        )

        self.max_ep_len = max_ep_len
        self.noise = noise
        self.noise_std = noise_std
        self.value_coeff = kwargs.get("value_coeff", 0.5)
        self.entropy_coeff = kwargs.get("entropy_coeff", 0.01)

        self.empty_logs()
        self.create_model()

    def create_model(self) -> None:
        """create actor critic model and initialize optimizers
        """
        input_dim, action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network_type
        )

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac = get_model("ac", self.network_type)(
            input_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

        self.rollout = RolloutBuffer(self.rollout_size, self.env)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """select action using policy, given state

        Args:
            state (np.ndarray): environment state
            deterministic (bool): True if deterministic policy. False if stochastic

        Returns:
            np.ndarray: action
            np.ndarray: value of state
            np.ndarray: categorical log probability of chosen action
        """
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on actor output
        action, dist = self.ac.get_action(state, deterministic=False)
        value = self.ac.get_value(state)

        return action.detach().cpu().numpy(), value, dist.log_prob(action).cpu()

    def get_traj_loss(self, value, done) -> None:
        """calculate discounted rewards and losses over single trajectory

        Args:
            value (np.ndarray): values of states in trajectory
            done (list(bool)): If each state in trajectory is a terminal state or not. True if terminal.
                False if not.
        """
        self.rollout.compute_returns_and_advantage(values.detach().cpu().numpy(), dones)

    def get_value_log_probs(self, state, action):
        """calculate value of state and log probability of action

        Args:
            state (np.ndarray): environment state
            action (np.ndarray): agent's action

        Returns:
            np.ndarray: value of state
            np.ndarray: log probability of taking action at state according to policy
        """

        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state)
        return val, c.log_prob(action)

    def update_policy(self) -> None:
        """update parameters of actor and critic networks
        """

        for rollout in self.rollout.get(self.batch_size):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.advantages * log_prob
            policy_loss = -torch.mean(policy_loss)
            self.logs["policy_loss"].append(policy_loss.item())

            value_loss = self.value_coeff * F.mse_loss(rollout.returns, values.cpu())
            self.logs["value_loss"].append(torch.mean(value_loss).item())

            entropy_loss = (torch.exp(log_prob) * log_prob).sum()
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
        """Returns important hyperparameters that need to be loaded or saved

        Returns:
            dict: hyperparameters that need to be loaded or saved
        """
        hyperparams = {
            "network_type": self.network_type,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "rollout_size": self.rollout_size,
            "policy_weights": self.ac.actor.state_dict(),
            "value_weights": self.ac.critic.state_dict(),
        }

        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.actor.load_state_dict(weights["policy_weights"])
        self.ac.critic.load_state_dict(weights["value_weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """Returns logging parameters for monitoring training

        Returns:
            dict: Logging parameters for monitoring training
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
        """Empties log dictionaries for policy loss, value loss and entropy
        """

        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["policy_entropy"] = []
        self.rewards = []
