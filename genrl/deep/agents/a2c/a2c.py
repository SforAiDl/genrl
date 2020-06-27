from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

from ....environments.vec_env import VecEnv
from ...common import OnPolicyAgent, RolloutBuffer, get_env_properties, get_model


class A2C(OnPolicyAgent):
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

        self.create_model()

    def create_model(self) -> None:
        """
        Creates actor critic model and initialises optimizers
        """
        (state_dim, action_dim, discrete, action_lim) = get_env_properties(self.env)

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        self.actor_optimizer = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)

        self.critic_optimizer = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

        self.rollout = RolloutBuffer(
            self.rollout_size,
            self.env.observation_space,
            self.env.action_space,
            n_envs=self.env.n_envs,
        )

        # load paramaters if already trained
        if self.run_num is not None:
            self.load(self)
            self.ac.actor.load_state_dict(self.checkpoint["actor_weights"])
            self.ac.critic.load_state_dict(self.checkpoint["critic_weights"])
            for key, item in self.checkpoint.items():
                if key not in ["actor_weights", "critic_weights"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Selection of action

        :param state: Observation state
        :param deterministic: Action selection type
        :type state: int, float, ...
        :type deterministic: bool
        :returns: Action based on the state and epsilon value
        :rtype: int, float, ...
        """
        state = Variable(torch.as_tensor(state).float().to(self.device))

        # create distribution based on policy_fn output
        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state).unsqueeze(0)

        return a, val, c.log_prob(a)

    def get_traj_loss(self, value, done) -> None:
        """
        (Get trajectory of agent to calculate discounted rewards and
calculate losses)
        """
        self.rollout.compute_returns_and_advantage(value.detach().cpu().numpy(), done)

    def get_value_log_probs(self, state, action):
        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state)
        return val, c.log_prob(action)

    def update_policy(self) -> None:

        for rollout in self.rollout.get(self.batch_size):

            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.advantages * log_prob
            policy_loss = -torch.sum(policy_loss)

            value_loss = self.value_coeff * F.mse_loss(rollout.returns, values)

            entropy_loss = (torch.exp(log_prob) * log_prob).sum()

            actor_loss = policy_loss + self.entropy_coeff * entropy_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
            self.critic_optimizer.step()

    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Loads important hyperparameters that need to be loaded or saved

        :returns: Hyperparameters that need to be saved or loaded
        :rtype: dict
        """
        hyperparams = {
            "network_type": self.network_type,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "gamma": self.gamma,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "actor_weights": self.ac.actor.state_dict(),
            "critic_weights": self.ac.critic.state_dict(),
        }

        return hyperparams
