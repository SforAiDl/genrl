import numpy as np
import torch
import torch.nn as nn
import gym
from copy import deepcopy

from ...common import (
    ReplayBuffer,
    get_model,
    save_params,
    load_params,
    get_env_properties,
    set_seeds,
    venv,
)
from typing import Tuple, Union, Dict, Optional, Any


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
    :param save_interval: (int) Number of steps between saves of models
    :param layers: (tuple or list) Number of neurons in hidden layers
    :param tensorboard_log: (str) the log location for tensorboard (if None,
        no logging)
    :param seed (int): seed for torch and gym
    :param render (boolean): if environment is to be rendered
    :param device (str): device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :param run_num: (boolean) if model has already been trained
    :param save_name: (str) model save name (if None, model hasn't been
        pretrained)
    :param save_version: (int) model save version (if None, model hasn't been
        pretrained)
    :param run_num: model run number if it has already been trained
    :param save_model: model save directory
    :param load_model: model loading path
    :type run_num: int
    :type save_model: string
    :type load_model: string
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, venv],
        gamma: float = 0.99,
        replay_size: int = 1000000,
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
        tensorboard_log: str = None,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 5000,
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
        self.save_interval = save_interval
        self.layers = layers
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save = save_params
        self.load = load_params

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # Setup tensorboard writer
        self.writer = None
        if self.tensorboard_log is not None:  # pragma: no cover
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.create_model()
        self.checkpoint = self.get_hyperparams()

    def create_model(self) -> None:
        state_dim, action_dim, discrete, _ = get_env_properties(self.env)
        if discrete == True:
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

        if self.load_model is not None:
            self.load(self)
            self.ac.actor.load_state_dict(self.checkpoint["policy_weights"])
            self.ac.qf1.load_state_dict(self.checkpoint["q1_weights"])
            self.ac.qf2.load_state_dict(self.checkpoint["q2_weights"])

            for key, item in self.checkpoint.items():
                if key not in ["weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

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

    def select_action(
        self, state: np.ndarray, deterministic: bool = True
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
        reward: float,
        next_state: np.ndarray,
        done: bool,
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

    def update_params(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        timestep: int,
    ) -> None:
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

        for t in range(0, total_steps, self.env.n_envs):
            # execute single transition
            if t > self.start_steps:
                action = self.select_action(state, deterministic=True)
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
                            sum(episode), np.mean(episode_reward), t
                        )
                    )

                for i, d in enumerate(done):
                    # print(d)
                    if d or episode_len[i] == self.max_ep_len:
                        episode_reward[i] = 0
                        episode_len[i] = 0
                        episode += 1

                # if self.noise is not None:
                #     self.noise.reset()

                # if self.tensorboard_log:
                #     self.writer.add_scalar("episode_reward", np.mean(episode_reward), t)

                # state, episode_reward, episode_len = self.env.reset(), np.zeros(self.env.n_envs), np.zeros(self.env.n_envs)
                # episode += 1

            # update params
            if t >= self.start_update and t % self.update_interval == 0:
                for _ in range(self.update_interval):
                    batch = self.replay_buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = (
                        x.to(self.device) for x in batch
                    )
                    # print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
                    self.update_params(
                        states, actions, rewards.unsqueeze(1), next_states, dones, t
                    )

            if self.save_model is not None:
                if t >= self.start_update and t % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, t)
                    print("Saved current model")

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

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
            "policy_weights": self.ac.actor.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    algo = TD3("mlp", env)
    algo.learn()
