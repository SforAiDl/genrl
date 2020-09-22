from typing import Any, Dict

import gym
import torch
import torch.optim as opt

from genrl.agents import OnPolicyAgent
from genrl.utils import (
    compute_returns_and_advantage,
    get_env_properties,
    get_model,
    safe_mean,
)


class VPG(OnPolicyAgent):
    """
    Vanilla Policy Gradient algorithm

    Paper https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf

        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        rollout_size (int): Capacity of the Rollout Buffer
        buffer_type (str): Choose the type of Buffer: ["rollout"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, **kwargs):
        super(VPG, self).__init__(*args, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self):
        """Initialize policy network"""
        state_dim, action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network
        )
        if isinstance(self.network, str):
            # Instantiate networks and optimizers
            self.actor = get_model("p", self.network)(
                state_dim,
                action_dim,
                self.policy_layers,
                "V",
                discrete,
                action_lim=action_lim,
            ).to(self.device)
        else:
            self.actor = self.network.to(self.device)

        self.optimizer_policy = opt.Adam(self.actor.parameters(), lr=self.lr_policy)

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action given state

        Action Selection for Vanilla Policy Gradient

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
            value (:obj:`torch.Tensor`): Value of given state. In VPG, there is no critic
                to find the value so we set this to a default 0 for convenience
            log_prob (:obj:`torch.Tensor`): Log probability of selected action
        """
        # create distribution based on policy_fn output
        action, dist = self.actor.get_action(state, deterministic=deterministic)

        return (
            action.detach(),
            torch.zeros((1, self.env.n_envs)),
            dist.log_prob(action).cpu(),
        )

    def get_log_probs(self, states: torch.Tensor, actions: torch.Tensor):
        """Get log probabilities of action values

        Actions taken by actor and their respective states are analysed to get
        log probabilities

        Args:
            states (:obj:`torch.Tensor`): States encountered in rollout
            actions (:obj:`torch.Tensor`): Actions taken in response to respective states

        Returns:
            log_probs (:obj:`torch.Tensor`): Log of action probabilities given a state
        """
        states, actions = states.to(self.device), actions.to(self.device)
        _, dist = self.actor.get_action(states, deterministic=False)
        return dist.log_prob(actions).cpu()

    def get_traj_loss(self, values, dones):
        """Get loss from trajectory traversed by agent during rollouts

        Computes the returns and advantages needed for calculating loss

        Args:
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            dones (:obj:`list` of bool): Game over statuses of each environment
        """
        compute_returns_and_advantage(
            self.rollout, values.detach().cpu().numpy(), dones.cpu().numpy()
        )

    def update_params(self) -> None:
        """Updates the the A2C network

        Function to update the A2C actor-critic architecture
        """
        for rollout in self.rollout.get(self.batch_size):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            log_prob = self.get_log_probs(rollout.observations, actions)

            loss = rollout.returns * log_prob

            loss = -torch.mean(loss)
            self.logs["loss"].append(loss.item())

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_policy.step()

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
            "lr_policy": self.lr_policy,
            "rollout_size": self.rollout_size,
        }

        return hyperparams, self.actor.state_dict()

    def _load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`dict`): Dictionary of different neural net weights
        """
        self.actor.load_state_dict(weights)

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "loss": safe_mean(self.logs["loss"]),
            "mean_reward": safe_mean(self.rewards),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs"""
        self.logs = {}
        self.logs["loss"] = []
        self.rewards = []
