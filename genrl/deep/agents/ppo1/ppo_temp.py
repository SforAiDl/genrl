from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from ....environments import VecEnv
from ...common import OnPolicyAgent, RolloutBuffer, get_env_properties, get_model


class PPO1(OnPolicyAgent):
    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        batch_size: int = 256,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        actor_batch_size: int = 64,
        epochs: int = 1000,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        layers: Tuple = (64, 64),
        policy_copy_interval: int = 20,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
        rollout_size: int = 2048,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
    ):

        super(PPO1, self).__init__(
            network_type,
            env,
            batch_size,
            layers,
            gamma,
            lr_policy,
            lr_value,
            actor_batch_size,
            epochs,
            seed,
            render,
            device,
            run_num,
            save_model,
            load_model,
            save_interval,
            rollout_size,
        )

        self.clip_param = clip_param
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff

        self.create_model()

    def create_model(self):
        # Instantiate networks and optimizers
        state_dim, action_dim, disc, action_lim = get_env_properties(self.env)
        self.policy_new = get_model("p", self.network_type)(
            state_dim, action_dim, self.layers, disc=disc, action_lim=action_lim
        )
        self.policy_new = self.policy_new.to(self.device)

        self.value_fn = get_model("v", self.network_type)(state_dim, action_dim).to(
            self.device
        )

        # load paramaters if already trained
        if self.load_model is not None:
            self.load(self)
            self.policy_new.load_state_dict(self.checkpoint["policy_weights"])
            self.value_fn.load_state_dict(self.checkpoint["value_weights"])
            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.optimizer_policy = opt.Adam(
            self.policy_new.parameters(), lr=self.lr_policy
        )
        self.optimizer_value = opt.Adam(self.value_fn.parameters(), lr=self.lr_value)

        self.rollout = RolloutBuffer(
            self.rollout_size,
            self.env.observation_space,
            self.env.action_space,
            n_envs=self.env.n_envs,
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state).float().to(self.device)
        # create distribution based on policy output
        action, c_new = self.policy_new.get_action(state, deterministic=False)
        value = self.value_fn.get_value(state)

        return action.detach().cpu().numpy(), value, c_new.log_prob(action)

    def evaluate_actions(self, obs, old_actions):
        value = self.value_fn.get_value(obs)
        _, dist = self.policy_new.get_action(obs)
        return value, dist.log_prob(old_actions), dist.entropy()

    def get_traj_loss(self, values, dones):
        self.rollout.compute_returns_and_advantage(
            values.detach().cpu().numpy(), dones, use_gae=True
        )

    def update_policy(self):

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
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            values = values.flatten()

            value_loss = nn.functional.mse_loss(rollout.returns, values)

            entropy_loss = -torch.mean(entropy)  # Change this to entropy

            loss = (
                policy_loss + self.entropy_coeff * entropy_loss
            )  # + self.vf_coef * value_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), 0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), 0.5)
            self.optimizer_value.step()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "actor_batch_size": self.actor_batch_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "policy_weights": self.policy_new.state_dict(),
            "value_weights": self.value_fn.state_dict(),
        }

        return hyperparams
