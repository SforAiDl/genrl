from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from ...common import (
    RolloutBuffer,
    get_env_properties,
    get_model,
    load_params,
    save_params,
    set_seeds,
    venv,
)


class PPO1:
    """
    Proximal Policy Optimization algorithm (Clipped policy).

    Paper: https://arxiv.org/abs/1707.06347

    :param network_type: The deep neural network layer types ['mlp']
    :param env: The environment to learn from
    :param timesteps_per_actorbatch: timesteps per actor per update
    :param gamma: discount factor
    :param clip_param: clipping parameter epsilon
    :param actor_batchsize: trajectories per optimizer epoch
    :param epochs: the optimizer's number of epochs
    :param lr_policy: policy network learning rate
    :param lr_value: value network learning rate
    :param policy_copy_interval: number of optimizer before copying
        params from new policy to old policy
    :param save_interval: Number of episodes between saves of models
    :param seed: seed for torch and gym
    :param device: device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :param run_num: if model has already been trained
    :param save_model: directory the user wants to save models to
    :param load_model: model loading path
    :type network_type: str
    :type env: Gym environment
    :type timesteps_per_actorbatch: int
    :type gamma: float
    :type clip_param: float
    :type actor_batchsize: int
    :type epochs: int
    :type lr_policy: float
    :type lr_value: float
    :type policy_copy_interval: int
    :type save_interval: int
    :type seed: int
    :type device: string
    :type run_num: boolean
    :type save_model: string
    :type load_model: string
    :type rollout_size: int
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, venv],
        timesteps_per_actorbatch: int = 256,
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
    ):
        self.network_type = network_type
        self.env = env
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.gamma = gamma
        self.clip_param = clip_param
        self.actor_batch_size = actor_batch_size
        self.epochs = epochs
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.layers = layers
        self.seed = seed
        self.render = render
        self.policy_copy_interval = policy_copy_interval
        self.save_interval = save_interval
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save = save_params
        self.load = load_params
        self.rollout_size = rollout_size

        self.ent_coef = 0.01
        self.vf_coef = 0.5

        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["entropy_loss"] = []

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        self.create_model()

    def create_model(self) -> None:
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

    # get clipped loss for single trajectory (episode)
    def get_traj_loss(self, values, dones):
        self.rollout.compute_returns_and_advantage(
            values.detach().cpu().numpy(), dones, use_gae=True
        )

    def update_policy(self):

        for rollout in self.rollout.get(256):
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
            self.logs["policy_loss"].append(policy_loss.item())

            values = values.flatten()

            value_loss = nn.functional.mse_loss(rollout.returns, values)
            self.logs["value_loss"].append(value_loss.item())

            entropy_loss = -torch.mean(entropy)  # Change this to entropy
            self.logs["policy_entropy"].append(entropy_loss.item())

            loss = (
                policy_loss + self.ent_coef * entropy_loss
            )  # + self.vf_coef * value_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), 0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), 0.5)
            self.optimizer_value.step()

    def collect_rollouts(self, state):

        for i in range(2048):

            with torch.no_grad():
                action, values, old_log_probs = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(np.array(action))
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                dones,
                values,
                old_log_probs,
            )

            state = next_state

            for i, done in enumerate(dones):
                if done:
                    self.rewards.append(self.epoch_reward[i])
                    self.epoch_reward[i] = 0

        return values, dones

    def learn(self):  # pragma: no cover
        # training loop
        state = self.env.reset()
        for epoch in range(self.epochs):
            self.epoch_reward = np.zeros(self.env.n_envs)

            self.rollout.reset()
            self.rewards = []

            values, done = self.collect_rollouts(state)

            self.get_traj_loss(values, done)

            self.update_policy()

            if epoch % 1 == 0:
                print("Episode: {}, reward: {}".format(epoch, np.mean(self.rewards)))
                self.rewards = []

            if self.save_model is not None:
                if epoch % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, epoch)
                    print("Saved current model")

        self.env.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "actor_batch_size": self.actor_batch_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "policy_weights": self.policy_new.state_dict(),
            "value_weights": self.value_fn.state_dict(),
        }

        return hyperparams

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """

        logs = {
            "policy_loss": np.mean(self.logs["policy_loss"]),
            "value_loss": np.mean(self.logs["value_loss"]),
            "policy_entropy": np.mean(self.logs["entropy_loss"]),
            "mean_reward": np.mean(self.reward),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """

        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["policy_entropy"] = []
        self.rewards = []


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    algo = PPO1("mlp", env)
    algo.learn()
