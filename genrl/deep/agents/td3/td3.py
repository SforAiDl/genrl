from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from genrl.deep.common import (
    ReplayBuffer,
    get_env_properties,
    get_model,
    safe_mean,
    set_seeds,
)
from genrl.environments import VecEnv


class TD3:
    """
    Twin Delayed DDPG

    Paper: https://arxiv.org/abs/1509.02971

    :param network_type: (str) The deep neural network layer types ['mlp']
    :param env: (Gym environment) The environment to learn from
    :param gamma: (float) discount factor
    :param replay_size: (int) Replay memory size
    :param batch_size: (int) Update batch size
    :param lr_p: (float) Policy network learning rate
    :param lr_q: (float) Q network learning rate
    :param polyak: (float) Polyak averaging weight to update target network
    :param policy_frequency: (int) Update actor and target networks every
        policy_frequency steps
    :param epochs: (int) Number of epochs
    :param start_steps: (int) Number of exploratory steps at start
    :param steps_per_epoch: (int) Number of steps per epoch
    :param noise_std: (float) Standard deviation for action noise
    :param max_ep_len: (int) Maximum steps per episode
    :param start_update: (int) Number of steps before first parameter update
    :param update_interval: (int) Number of steps between parameter updates
    :param layers: (tuple or list) Number of neurons in hidden layers
    :param seed (int): seed for torch and gym
    :param render (boolean): if environment is to be rendered
    :param device (str): device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :type network_type: str
    :type env: Gym environment
    :type gamma: float
    :type replay_size: int
    :type batch_size: int
    :type lr_p: float
    :type lr_q: float
    :type polyak: float
    :type policy_frequency: int
    :type epochs: int
    :type start_steps: int
    :type steps_per_epoch: int
    :type noise_std: float
    :type max_ep_len: int
    :type start_update: int
    :type update_interval: int
    :type layers: tuple or list
    :type seed: int
    :type render: boolean
    :type device: str
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        gamma: float = 0.99,
        replay_size: int = 1000,
        batch_size: int = 100,
        lr_p: float = 0.001,
        lr_q: float = 0.001,
        polyak: float = 0.995,
        policy_frequency: int = 2,
        epochs: int = 100,
        start_steps: int = 10000,
        steps_per_epoch: int = 4000,
        noise: Optional[Any] = None,
        noise_std: float = 0.1,
        max_ep_len: int = 1000,
        start_update: int = 1000,
        update_interval: int = 50,
        layers: Tuple = (256, 256),
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
    ):

        self.network_type = network_type
        self.env = env
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.polyak = polyak
        self.policy_frequency = policy_frequency
        self.epochs = epochs
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
        self.noise = noise
        self.noise_std = noise_std
        self.max_ep_len = max_ep_len
        self.start_update = start_update
        self.update_interval = update_interval
        self.layers = layers
        self.seed = seed
        self.render = render

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        self.empty_logs()
        self.create_model()

    def create_model(self) -> None:
        state_dim, action_dim, discrete, _ = get_env_properties(self.env)
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )
        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "Qsa", False
        ).to(self.device)

        self.ac.qf1 = self.ac.critic
        self.ac.qf2 = get_model("v", self.network_type)(
            state_dim, action_dim, hidden=self.layers, val_type="Qsa"
        )

        self.ac.qf1.to(self.device)
        self.ac.qf2.to(self.device)

        self.ac_target = deepcopy(self.ac).to(self.device)

        # freeze target network params
        for param in self.ac_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.replay_size, self.env)
        self.q_params = list(self.ac.qf1.parameters()) + list(self.ac.qf2.parameters())
        self.optimizer_q = torch.optim.Adam(self.q_params, lr=self.lr_q)

        self.optimizer_policy = torch.optim.Adam(
            self.ac.actor.parameters(), lr=self.lr_p
        )

    def update_params_before_select_action(self, timestep: int) -> None:
        """
        Update any parameters before selecting action like epsilon for decaying epsilon greedy

        :param timestep: Timestep in the training process
        :type timestep: int
        """
        pass

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action = self.ac_target.get_action(
                torch.as_tensor(state, dtype=torch.float32, device=self.device),
                deterministic=deterministic,
            )[0].numpy()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return np.clip(
            action, -self.env.action_space.high[0], self.env.action_space.high[0]
        )

    def get_q_loss(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> torch.Tensor:
        q1 = self.ac.qf1.get_value(torch.cat([state, action], dim=-1))
        q2 = self.ac.qf2.get_value(torch.cat([state, action], dim=-1))

        with torch.no_grad():
            target_q1 = self.ac_target.qf1.get_value(
                torch.cat(
                    [
                        next_state,
                        self.ac_target.get_action(next_state, deterministic=True)[0],
                    ],
                    dim=-1,
                )
            )
            target_q2 = self.ac_target.qf2.get_value(
                torch.cat(
                    [
                        next_state,
                        self.ac_target.get_action(next_state, deterministic=True)[0],
                    ],
                    dim=-1,
                )
            )
            target_q = torch.min(target_q1, target_q2).unsqueeze(1)

            target = reward.squeeze(1) + self.gamma * (1 - done) * target_q.squeeze(1)

        l1 = nn.MSELoss()(q1, target)
        l2 = nn.MSELoss()(q2, target)

        return l1 + l2

    def get_p_loss(self, state: np.array) -> torch.Tensor:
        q_pi = self.ac.get_value(
            torch.cat([state, self.ac.get_action(state, deterministic=True)[0]], dim=-1)
        )
        return -torch.mean(q_pi)

    def update_params(self, update_interval: int) -> None:
        for timestep in range(update_interval):
            batch = self.replay_buffer.sample(self.batch_size)
            state, action, reward, next_state, done = (x.to(self.device) for x in batch)
            self.optimizer_q.zero_grad()
            # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
            loss_q = self.get_q_loss(state, action, reward, next_state, done)
            loss_q.backward()
            self.optimizer_q.step()

            # Delayed Update
            if timestep % self.policy_frequency == 0:
                # freeze critic params for policy update
                for param in self.q_params:
                    param.requires_grad = False

                self.optimizer_policy.zero_grad()
                loss_p = self.get_p_loss(state)
                loss_p.backward()
                self.optimizer_policy.step()

                # unfreeze critic params
                for param in self.ac.critic.parameters():
                    param.requires_grad = True

                # update target network
                with torch.no_grad():
                    for param, param_target in zip(
                        self.ac.parameters(), self.ac_target.parameters()
                    ):
                        param_target.data.mul_(self.polyak)
                        param_target.data.add_((1 - self.polyak) * param.data)

                self.logs["policy_loss"].append(loss_p.item())
                self.logs["value_loss"].append(loss_q.item())

    def learn(self) -> None:  # pragma: no cover
        state, episode_reward, episode_len, episode = (
            self.env.reset(),
            np.zeros(self.env.n_envs),
            np.zeros(self.env.n_envs),
            np.zeros(self.env.n_envs),
        )
        total_steps = self.steps_per_epoch * self.epochs * self.env.n_envs

        if self.noise is not None:
            self.noise.reset()

        for timestep in range(0, total_steps, self.env.n_envs):
            # execute single transition
            if timestep > self.start_steps:
                action = self.select_action(state)
            else:
                action = self.env.sample()

            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            episode_reward += reward
            episode_len += 1

            # dont set d to True if max_ep_len reached
            # done = self.env.n_envs*[False] if np.any(episode_len == self.max_ep_len) else done
            done = np.array(
                [
                    False if episode_len[i] == self.max_ep_len else done[i]
                    for i, ep_len in enumerate(episode_len)
                ]
            )

            self.replay_buffer.extend(zip(state, action, reward, next_state, done))

            state = next_state

            if np.any(done) or np.any(episode_len == self.max_ep_len):

                if sum(episode) % 20 == 0:
                    print(
                        "Ep: {}, reward: {}, t: {}".format(
                            sum(episode), np.mean(episode_reward), timestep
                        )
                    )

                for i, di in enumerate(done):
                    # print(d)
                    if di or episode_len[i] == self.max_ep_len:
                        episode_reward[i] = 0
                        episode_len[i] = 0
                        episode += 1

                if self.noise is not None:
                    self.noise.reset()

                state, episode_reward, episode_len = (
                    self.env.reset(),
                    np.zeros(self.env.n_envs),
                    np.zeros(self.env.n_envs),
                )
                episode += 1

            # update params
            if timestep >= self.start_update and timestep % self.update_interval == 0:
                self.update_params(self.update_interval)

        self.env.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "gamma": self.gamma,
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "polyak": self.polyak,
            "policy_frequency": self.policy_frequency,
            "noise_std": self.noise_std,
            "q1_weights": self.ac.qf1.state_dict(),
            "q2_weights": self.ac.qf2.state_dict(),
            "actor_weights": self.ac.actor.state_dict(),
        }

        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.actor.load_state_dict(weights["actor_weights"])
        self.ac.qf1.load_state_dict(weights["q1_weights"])
        self.ac.qf2.load_state_dict(weights["q2_weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
