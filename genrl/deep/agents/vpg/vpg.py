from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.optim as opt

from genrl.deep.agents.base import OnPolicyAgent
from genrl.deep.common.base import BasePolicy
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean
from genrl.environments.vec_env import VecEnv


class VPG(OnPolicyAgent):
    """
    Vanilla Policy Gradient algorithm

    Paper https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf

    :param network: The deep neural network layer types ['mlp']
    :param env: The environment to learn from
    :param timesteps_per_actorbatch: timesteps per actor per update
    :param gamma: discount factor
    :param actor_batchsize: trajectories per optimizer epoch
    :param epochs: the optimizer's number of epochs
    :param lr_policy: policy network learning rate
    :param seed: seed for torch and gym
    :param device: device to use for tensor operations; \
'cpu' for cpu and 'cuda' for gpu
    :param load_model: model loading path
    :param rollout_size: Rollout Buffer Size
    :type network: str or BaseActorCritic
    :type env: Gym environment(s)
    :type timesteps_per_actorbatch: int
    :type gamma: float
    :type actor_batchsize: int
    :type epochs: int
    :type lr_policy: float
    :type seed: int
    :type device: str
    :type load_model: string
    :type rollout_size: int
    """

    def __init__(self, *args, **kwargs):
        super(VPG, self).__init__(*args, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self):
        """Initialize policy network
        """
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
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
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
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on policy_fn output
        action, dist = self.actor.get_action(state, deterministic=deterministic)

        return (
            action.detach().cpu().numpy(),
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
        self.rollout.compute_returns_and_advantage(values.detach().cpu().numpy(), dones)

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
        """
        hyperparams = {
            "network": self.network,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "lr_policy": self.lr_policy,
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
            "loss": safe_mean(self.logs["loss"]),
            "mean_reward": safe_mean(self.rewards),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs
        """
        self.logs = {}
        self.logs["loss"] = []
        self.rewards = []
