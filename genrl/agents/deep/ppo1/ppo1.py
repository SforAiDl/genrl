from typing import Any, Dict

import gym
import torch  # noqa
import torch.nn as nn  # noqa
import torch.optim as opt  # noqa

from genrl.agents import OnPolicyAgent
from genrl.utils import (
    compute_returns_and_advantage,
    get_env_properties,
    get_model,
    safe_mean,
)


class PPO1(OnPolicyAgent):
    """
    Proximal Policy Optimization algorithm (Clipped policy).

    Paper: https://arxiv.org/abs/1707.06347

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        shared_layers(:obj:`tuple` of :obj:`int`): Sizes of shared layers in Actor Critic if using
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        rollout_size (int): Capacity of the Rollout Buffer
        buffer_type (str): Choose the type of Buffer: ["rollout"]
        clip_param (float): Epsilon for clipping policy loss
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
        clip_param: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        **kwargs
    ):
        super(PPO1, self).__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.activation = kwargs["activation"] if "activation" in kwargs else "relu"

        self.empty_logs()

        if self.create_model:
            self._create_model()

    def _create_model(self):
        """Function to initialize Actor-Critic architecture

        This will create the Actor-Critic net for the agent and initialise the action noise
        """
        # Instantiate networks and optimizers
        state_dim, action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network
        )
        if isinstance(self.network, str):
            arch = self.network
            if self.shared_layers is not None:
                arch += "s"
            self.ac = get_model("ac", arch)(
                state_dim,
                action_dim,
                shared_layers=self.shared_layers,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_typ="V",
                discrete=discrete,
                action_lim=action_lim,
                activation=self.activation,
            ).to(self.device)
        else:
            self.ac = self.network.to(self.device)

        actor_params, critic_params = self.ac.get_params()
        self.optimizer_policy = opt.Adam(actor_params, lr=self.lr_policy)
        self.optimizer_value = opt.Adam(critic_params, lr=self.lr_value)

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
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
        # create distribution based on policy output
        action, dist = self.ac.get_action(state, deterministic=deterministic)
        value = self.ac.get_value(state)

        return action.detach(), value, dist.log_prob(action).cpu()

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

    def get_traj_loss(self, values, dones):
        """Get loss from trajectory traversed by agent during rollouts

        Computes the returns and advantages needed for calculating loss

        Args:
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            dones (:obj:`list` of bool): Game over statuses of each environment
        """
        compute_returns_and_advantage(
            self.rollout,
            values.detach().cpu().numpy(),
            dones.cpu().numpy(),
            use_gae=True,
        )

    def update_params(self):
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

            advantages = rollout.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_prob - rollout.old_log_prob)

            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(
                ratio, 1 - self.clip_param, 1 + self.clip_param
            )
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            self.logs["policy_loss"].append(policy_loss.item())

            values = values.flatten()

            value_loss = self.value_coeff * nn.functional.mse_loss(
                rollout.returns, values.cpu()
            )
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
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "network": self.network,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "rollout_size": self.rollout_size,
        }

        return hyperparams, self.ac.state_dict()

    def _load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`dict`): Dictionary of different neural net weights
        """
        self.ac.load_state_dict(weights)

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
