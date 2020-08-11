from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.optim as opt

from genrl.deep.agents.base import OnPolicyAgent
from genrl.deep.common import (
    BasePolicy,
    RolloutBuffer,
    get_env_properties,
    get_model,
    safe_mean,
)
from genrl.environments import VecEnv


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

    def __init__(
        self,
        network: Union[str, BasePolicy],
        env: Union[gym.Env, VecEnv],
        batch_size: int = 256,
        gamma: float = 0.99,
        lr_policy: float = 0.01,
        policy_layers: Tuple = (32, 32),
        rollout_size: int = 2048,
        **kwargs
    ):

        super(VPG, self).__init__(
            network,
            env,
            batch_size=batch_size,
            policy_layers=policy_layers,
            gamma=gamma,
            lr_policy=lr_policy,
            lr_value=None,
            rollout_size=rollout_size,
            **kwargs
        )

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self):
        """
        Initialize the actor and critic networks
        """
        if isinstance(self.network, str):
            input_dim, action_dim, discrete, action_lim = get_env_properties(
                self.env, self.network
            )

            # Instantiate networks and optimizers
            self.actor = get_model("p", self.network)(
                input_dim,
                action_dim,
                self.policy_layers,
                "V",
                discrete,
                action_lim=action_lim,
            ).to(self.device)
        else:
            self.actor = self.network.to(self.device)

        self.optimizer_policy = opt.Adam(self.actor.parameters(), lr=self.lr_policy)

        self.rollout = RolloutBuffer(self.rollout_size, self.env,)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action for the given state

        :param state: State for which action has to be sampled
        :param deterministic: Whether the action is deterministic or not
        :type state: int, float, ...
        :type deterministic: bool
        :returns: The action
        :rtype: int, float, ...
        """
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on policy_fn output
        action, dist = self.actor.get_action(state, deterministic=deterministic)

        return (
            action.detach().cpu().numpy(),
            torch.zeros((1, self.env.n_envs)),
            dist.log_prob(action).cpu(),
        )

    def get_value_log_probs(self, state, action):
        state, action = state.to(self.device), action.to(self.device)
        _, dist = self.actor.get_action(state, deterministic=False)
        return dist.log_prob(action).cpu()

    def get_traj_loss(self, values, dones):
        """
        Calculates the loss for the trajectory
        """
        self.rollout.compute_returns_and_advantage(values.detach().cpu().numpy(), dones)

    def update_policy(self) -> None:
        for rollout in self.rollout.get(self.batch_size):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            log_prob = self.get_value_log_probs(rollout.observations, actions)

            loss = rollout.returns * log_prob

            loss = -torch.mean(loss)
            self.logs["loss"].append(loss.item())

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_policy.step()

    def get_hyperparams(self) -> Dict[str, Any]:
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
            "loss": safe_mean(self.logs["loss"]),
            "mean_reward": safe_mean(self.rewards),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["loss"] = []
        self.rewards = []
