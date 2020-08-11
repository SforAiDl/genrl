from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt

from genrl.deep.common.actor_critic import BaseActorCritic
from genrl.deep.common.buffers import ReplayBuffer
from genrl.deep.common.utils import get_env_properties, get_model, safe_mean, set_seeds
from genrl.environments.vec_env import VecEnv


class DDPG:
    """
    Deep Deterministic Policy Gradient algorithm (DDPG)

    Paper: https://arxiv.org/abs/1509.02971

    :param network: The deep neural network layer types ['mlp', 'cnn'] or a CustomClass
    :param env: The environment to learn from
    :param gamma: discount factor
    :param replay_size: Replay memory size
    :param batch_size: Update batch size
    :param lr_p: learning rate for policy optimizer
    :param lr_q: learning rate for value fn optimizer
    :param polyak: polyak averaging weight for target network update
    :param epochs: Number of epochs
    :param start_steps: Number of exploratory steps at start
    :param steps_per_epoch: Number of steps per epoch
    :param noise_std: Standard deviation for action noise
    :param max_ep_len: Maximum steps per episode
    :param start_update: Number of steps before first parameter update
    :param update_interval: Number of steps between parameter updates
    :param layers: Number of neurons in hidden layers
    :param seed: seed for torch and gym
    :param render: if environment is to be rendered
    :param device: device to use for tensor operations; ['cpu','cuda']
    :type network: string
    :type env: Gym environment
    :type gamma: float
    :type replay_size: int
    :type batch_size: int
    :type lr_p: float
    :type lr_q: float
    :type polyak: float
    :type epochs: int
    :type start_steps: int
    :type steps_per_epoch: int
    :type noise_std: float
    :type max_ep_len: int
    :type start_update: int
    :type update_interval: int
    :type layers: tuple
    :type seed: int
    :type render: bool
    :type device: string
    """

    def __init__(
        self,
        network: Union[str, BaseActorCritic],
        env: Union[gym.Env, VecEnv],
        create_model: bool = True,
        gamma: float = 0.99,
        replay_size: int = 1000000,
        batch_size: int = 100,
        lr_p: float = 0.0001,
        lr_q: float = 0.001,
        polyak: float = 0.995,
        epochs: int = 100,
        start_steps: int = 10000,
        steps_per_epoch: int = 4000,
        noise: Optional[Any] = None,
        noise_std: float = 0.1,
        max_ep_len: int = 1000,
        start_update: int = 1000,
        update_interval: int = 50,
        layers: Tuple = (32, 32),
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
    ):

        self.network = network
        self.env = env
        self.create_model = create_model
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.polyak = polyak
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

        # Setup tensorboard writer
        self.writer = None

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        """
        Initialize the model
        Initializes optimizer and replay buffers as well.
        """
        if isinstance(self.network, str):
            state_dim, action_dim, discrete, _ = get_env_properties(self.env)
            assert not discrete, "Discrete Environments not supported for {}.".format(
                __class__.__name__
            )

            self.ac = get_model("ac", self.network)(
                state_dim, action_dim, self.layers, "Qsa", False
            ).to(self.device)
        else:
            self.ac = self.network.to(self.device)
            action_dim = self.network.action_dim

        self.ac_target = deepcopy(self.ac).to(self.device)
        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        # freeze target network params
        for param in self.ac_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.replay_size, self.env)
        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_p)
        self.optimizer_q = opt.Adam(self.ac.critic.parameters(), lr=self.lr_q)

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
        """
        Selection of action

        :param state: Observation state
        :param deterministic: Action selection type
        :type state: int, float, ...
        :type deterministic: bool
        :returns: Action based on the state and epsilon value
        :rtype: int, float, ...
        """
        with torch.no_grad():
            action, _ = self.ac.get_action(
                torch.as_tensor(state, dtype=torch.float32).to(self.device),
                deterministic=deterministic,
            )
            action = action.detach().cpu().numpy()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return np.clip(
            action, self.env.action_space.low[0], self.env.action_space.high[0]
        )

    def get_q_loss(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> torch.Tensor:
        """
        Computes loss for Q-Network

        :param state: environment observation
        :param action: agent action
        :param: reward: environment reward
        :param next_state: environment next observation
        :param done: if episode is over
        :type state: int, float, ...
        :type action: float
        :type: reward: float
        :type next_state: int, float, ...
        :type done: bool
        :returns: the Q loss value
        :rtype: float
        """
        quality = self.ac.critic.get_value(torch.cat([state, action], dim=-1))

        with torch.no_grad():
            q_pi_target = self.ac_target.get_value(
                torch.cat(
                    [next_state, self.ac_target.get_action(next_state, True)[0]], dim=-1
                )
            )
            target = reward + self.gamma * (1 - done) * q_pi_target

        value_loss = F.mse_loss(quality, target)
        self.logs["value_loss"].append(value_loss.item())
        return value_loss

    def get_p_loss(self, state: np.ndarray) -> torch.Tensor:
        """
        Computes policy loss

        :param state: Environment observation
        :type state: int, float, ...
        :returns: Policy loss
        :rtype: float
        """
        q_pi = self.ac.get_value(
            torch.cat([state, self.ac.get_action(state, True)[0]], dim=-1)
        )

        policy_loss = torch.mean(q_pi)
        self.logs["policy_loss"].append(policy_loss.item())

        return -policy_loss

    def update_params(self, update_interval: int) -> None:
        """
        Takes the step for optimizer.

        :param timestep: timestep
        :type timestep: int
        """
        for timestep in range(update_interval):
            batch = self.replay_buffer.sample(self.batch_size)
            state, action, reward, next_state, done = (x.to(self.device) for x in batch)

            self.optimizer_q.zero_grad()
            loss_q = self.get_q_loss(state, action, reward, next_state, done)
            loss_q.backward()
            self.optimizer_q.step()

            # freeze critic params for policy update
            for param in self.ac.critic.parameters():
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

    def learn(self):  # pragma: no cover
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
            done = [
                False if ep_len == self.max_ep_len else done for ep_len in episode_len
            ]

            self.replay_buffer.extend(zip(state, action, reward, next_state, done))

            state = next_state

            if np.any(done) or np.any(episode_len == self.max_ep_len):

                if self.noise is not None:
                    self.noise.reset()

                if sum(episode) % 20 == 0:
                    print(
                        "Ep: {}, reward: {}, t: {}".format(
                            sum(episode), np.mean(episode_reward), timestep
                        )
                    )

                for i, di in enumerate(done):
                    if di:
                        episode_reward[i] = 0
                        episode_len[i] = 0
                        episode += 1

            # update params
            if timestep >= self.start_update and timestep % self.update_interval == 0:
                self.update_params(self.update_interval)

        self.env.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network": self.network,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_size": self.replay_size,
            "polyak": self.polyak,
            "noise_std": self.noise_std,
            "lr_policy": self.lr_p,
            "lr_value": self.lr_q,
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


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    algo = DDPG("mlp", env)
    algo.learn()
    algo.evaluate(algo)
