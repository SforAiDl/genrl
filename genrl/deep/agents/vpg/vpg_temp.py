from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.optim as opt
from torch.autograd import Variable

from ....environments import VecEnv
from ...common import OnPolicyAgent, RolloutBuffer, get_env_properties, get_model


class VPG(OnPolicyAgent):
    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        batch_size: int = 256,
        gamma: float = 0.99,
        epochs: int = 1000,
        lr_policy: float = 0.01,
        lr_value: float = 0.0005,
        policy_copy_interval: int = 20,
        layers: Tuple = (32, 32),
        rollout_size: int = 2048,
        **kwargs
    ):

        super(VPG, self).__init__(
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

        self.create_model()

    def create_model(self):
        """
        Initialize the actor and critic networks
        """
        state_dim, action_dim, discrete, action_lim = get_env_properties(self.env)
        # print(state_dim, action_dim, discrete)
        # print(self.load)
        # Instantiate networks and optimizers
        self.actor = get_model("p", self.network_type)(
            state_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        # load paramaters if already trained
        if self.load_model is not None:
            self.load(self)
            self.actor.load_state_dict(self.checkpoint["policy_weights"])

            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.optimizer_policy = opt.Adam(self.actor.parameters(), lr=self.lr_policy)

        self.rollout = RolloutBuffer(
            self.rollout_size,
            self.env.observation_space,
            self.env.action_space,
            n_envs=self.env.n_envs,
        )

        # print(self.actor)

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
        state = Variable(torch.as_tensor(state).float().to(self.device))

        # create distribution based on policy_fn output
        a, c = self.actor.get_action(state, deterministic=False)

        return a, c.log_prob(a), None

    def get_value_log_probs(self, state, action):
        a, c = self.actor.get_action(state, deterministic=False)
        return c.log_prob(action)

    def get_traj_loss(self, value, done):
        """
        Calculates the loss for the trajectory
        """
        self.rollout.compute_returns_and_advantage(value.detach().cpu().numpy(), done)

    def update_policy(self) -> None:

        for rollout in self.rollout.get(self.batch_size):

            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.returns * log_prob

            policy_loss = -torch.sum(policy_loss)

            loss = policy_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_policy.step()

    def collect_rollouts(self, initial_state):

        state = initial_state

        for i in range(self.rollout_size):

            action, old_log_probs, _ = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(action.numpy())
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                dones,
                torch.Tensor([0] * self.env.n_envs),
                old_log_probs.detach(),
            )

            state = next_state

            self.collect_rewards(dones)

        return torch.Tensor([0] * self.env.n_envs), dones

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "weights": self.ac.state_dict(),
        }

        return hyperparams
