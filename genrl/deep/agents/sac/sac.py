from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.distributions import Normal

from genrl.environments import VecEnv
from genrl.deep.common import ReplayBuffer, get_env_properties, get_model, safe_mean, set_seeds


class SAC:
    """
    Soft Actor Critic algorithm (SAC)

    Paper: https://arxiv.org/abs/1812.05905

    :param network_type: The deep neural network layer types ['mlp', 'cnn']
    :param env: The environment to learn from
    :param gamma: discount factor
    :param replay_size: Replay memory size
    :param batch_size: Update batch size
    :param lr: learning rate for optimizers
    :param alpha: entropy coefficient
    :param polyak: polyak averaging weight for target network update
    :param entropy_tuning: if alpha should be a learned parameter
    :param epochs: Number of epochs to train on
    :param start_steps: Number of initial exploratory steps
    :param steps_per_epoch: Number of parameter updates per epoch
    :param max_ep_len: Maximum number of steps per episode
    :param start_update: Number of steps before first parameter update
    :param update_interval: Number of step between updates
    :param layers: Neural network layer dimensions
    :param seed: seed for torch and gym
    :param render: if environment is to be rendered
    :param device: device to use for tensor operations; ['cpu','cuda']
    :type network_type: string
    :type env: Gym environment
    :type gamma: float
    :type replay_size: int
    :type batch_size: int
    :type lr: float
    :type alpha: float
    :type polyak: float
    :type entropy_tuning: bool
    :type epochs: int
    :type start_steps: int
    :type steps_per_epoch: int
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
        network_type: str,
        env: Union[gym.Env, VecEnv],
        gamma: float = 0.99,
        replay_size: int = 1000000,
        batch_size: int = 256,
        lr: float = 3e-4,
        alpha: float = 0.01,
        polyak: float = 0.995,
        entropy_tuning: bool = True,
        epochs: int = 1000,
        start_steps: int = 0,
        steps_per_epoch: int = 1000,
        max_ep_len: int = 1000,
        start_update: int = 256,
        update_interval: int = 1,
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
        self.lr = lr
        self.alpha = alpha
        self.polyak = polyak
        self.entropy_tuning = entropy_tuning
        self.epochs = epochs
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
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
        self.create_model()

    def create_model(self) -> None:
        """
        Initialize the model
        Initializes optimizer and replay buffers as well.
        """
        state_dim, action_dim, discrete, _ = get_env_properties(self.env)

        self.q1 = (
            get_model("v", self.network_type)(state_dim, action_dim, "Qsa", self.layers)
            .to(self.device)
            .float()
        )

        self.q2 = (
            get_model("v", self.network_type)(state_dim, action_dim, "Qsa", self.layers)
            .to(self.device)
            .float()
        )

        self.policy = (
            get_model("p", self.network_type)(
                state_dim, action_dim, self.layers, discrete, False, sac=True
            )
            .to(self.device)
            .float()
        )

        self.q1_targ = deepcopy(self.q1).to(self.device).float()
        self.q2_targ = deepcopy(self.q2).to(self.device).float()

        # freeze target parameters
        for param in self.q1_targ.parameters():
            param.requires_grad = False
        for param in self.q2_targ.parameters():
            param.requires_grad = False

        # optimizers
        self.q1_optimizer = opt.Adam(self.q1.parameters(), self.lr)
        self.q2_optimizer = opt.Adam(self.q2.parameters(), self.lr)
        self.policy_optimizer = opt.Adam(self.policy.parameters(), self.lr)

        if self.entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.replay_size, self.env)

        # set action scales
        if self.env.action_space is None:
            self.action_scale = torch.tensor(1.0).to(self.device)
            self.action_bias = torch.tensor(0.0).to(self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.0
            ).to(self.device)
            self.action_bias = torch.FloatTensor(
                (self.env.action_space.high + self.env.action_space.low) / 2.0
            ).to(self.device)

    def sample_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        sample action normal distribution parameterized by policy network

        :param state: Observation state
        :param deterministic: Is the greedy action being chosen?
        :type state: int, float, ...
        :type deterministic: bool
        :returns: action
        :returns: log likelihood of policy
        :returns: scaled mean of normal distribution
        :rtype: int, float, ...
        :rtype: float
        :rtype: float
        """
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()

        # reparameterization trick
        distribution = Normal(mean, std)
        xi = distribution.rsample()
        yi = torch.tanh(xi)
        action = yi * self.action_scale + self.action_bias
        log_pi = distribution.log_prob(xi)

        # enforcing action bound (appendix of paper)
        log_pi -= torch.log(
            self.action_scale * (1 - yi.pow(2)) + np.finfo(np.float32).eps
        )
        log_pi = log_pi.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.float(), log_pi, mean

    def update_params_before_select_action(self, timestep: int) -> None:
        """
        Update any parameters before selecting action like epsilon for decaying epsilon greedy

        :param timestep: Timestep in the training process
        :type timestep: int
        """
        pass

    def select_action(self, state, deterministic=False):
        """
        select action given a state

        :param state: Observation state
        :param deterministic: Is the greedy action being chosen?
        :type state: int, float, ...
        :type deterministic: bool
        """
        state = torch.FloatTensor(state).to(self.device)
        action, _, _ = self.sample_action(state, deterministic)
        return action.detach().cpu().numpy()

    def update_params(self, update_interval: int) -> (Tuple[float]):
        """
        Computes loss and takes optimizer step

        :param timestep: timestep
        :type timestep: int
        :returns: policy loss
        :rtype: float
        :returns: entropy coefficient loss
        :rtype: float
        """
        for timestep in range(update_interval):
            batch = self.replay_buffer.sample(self.batch_size)
            state, action, reward, next_state, done = (x.to(self.device) for x in batch)
            # compute targets
            if self.env.n_envs == 1:
                state, action, next_state = (
                    state.squeeze().float(),
                    action.squeeze(1).float(),
                    next_state.squeeze().float(),
                )
            else:
                state, action, next_state = (
                    state.reshape(-1, *self.env.obs_shape).float(),
                    action.reshape(-1, *self.env.action_shape).float(),
                    next_state.reshape(-1, *self.env.obs_shape).float(),
                )
                reward, done = reward.reshape(-1, 1), done.reshape(-1, 1)

            with torch.no_grad():
                next_action, next_log_pi, _ = self.sample_action(next_state)
                next_q1_targ = self.q1_targ(
                    torch.cat([next_state, next_action], dim=-1)
                )
                next_q2_targ = self.q2_targ(
                    torch.cat([next_state, next_action], dim=-1)
                )
                next_q_targ = (
                    torch.min(next_q1_targ, next_q2_targ) - self.alpha * next_log_pi
                )
                next_q = reward + self.gamma * (1 - done) * next_q_targ

            # compute losses
            q1 = self.q1(torch.cat([state, action], dim=-1))
            q2 = self.q2(torch.cat([state, action], dim=-1))

            q1_loss = nn.MSELoss()(q1, next_q)
            q2_loss = nn.MSELoss()(q2, next_q)

            pi, log_pi, _ = self.sample_action(state)
            q1_pi = self.q1(torch.cat([state, pi.float()], dim=-1).float())
            q2_pi = self.q2(torch.cat([state, pi.float()], dim=-1).float())
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

            # gradient step
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # alpha loss
            alpha_loss = torch.tensor(0.0).to(self.device)

            if self.entropy_tuning:
                alpha_loss = -(
                    self.log_alpha * (log_pi + self.target_entropy).detach()
                ).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()

            # soft update target params
            for target_param, param in zip(
                self.q1_targ.parameters(), self.q1.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * self.polyak + param.data * (1 - self.polyak)
                )

            for target_param, param in zip(
                self.q2_targ.parameters(), self.q2.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * self.polyak + param.data * (1 - self.polyak)
                )

        self.logs["q1_loss"].append(q1_loss.item())
        self.logs["q2_loss"].append(q2_loss.item())
        self.logs["policy_loss"].append(policy_loss.item())
        self.logs["alpha_loss"].append(alpha_loss.item())

    def learn(self) -> None:  # pragma: no cover

        total_steps = self.steps_per_epoch * self.epochs * self.env.n_envs

        episode_reward, episode_len = (
            np.zeros(self.env.n_envs),
            np.zeros(self.env.n_envs),
        )
        state = self.env.reset()
        for i in range(0, total_steps, self.env.n_envs):
            # done = [False] * self.env.n_envs

            # while not done:
            # sample action
            if i > self.start_steps:
                action = self.select_action(state)
            else:
                action = self.env.sample()

            if (
                i >= self.start_update
                and i % self.update_interval == 0
                and self.replay_buffer.pos > self.batch_size
            ):
                self.update_params(self.update_interval)

                # prepare transition for replay memory push
            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()

            done = [
                False if ep_len == self.max_ep_len else done for ep_len in episode_len
            ]

            if np.any(done) or np.any(episode_len == self.max_ep_len):
                for j, di in enumerate(done):
                    if di:
                        episode_reward[j] = 0
                        episode_len[j] = 0

            self.replay_buffer.extend(zip(state, action, reward, next_state, done))
            state = next_state

            if i > total_steps:
                break

            if sum(episode_len) % (5 * self.env.n_envs) == 0 and sum(episode_len) != 0:
                print(
                    "Episode: {}, total numsteps: {}, reward: {}".format(
                        sum(episode_len), i, episode_reward
                    )
                )
            # ep += 1

        self.env.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "gamma": self.gamma,
            "lr": self.lr,
            "replay_size": self.replay_size,
            "entropy_tuning": self.entropy_tuning,
            "alpha": self.alpha,
            "polyak": self.polyak,
            "q1_weights": self.q1.state_dict(),
            "q2_weights": self.q2.state_dict(),
            "policy_weights": self.policy.state_dict(),
        }

        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.q1.load_state_dict(weights["q1_weights"])
        self.q2.load_state_dict(weights["q2_weights"])
        self.policy.load_state_dict(weights["policy_weights"])

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """
        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "q1_loss": safe_mean(self.logs["q1_loss"]),
            "q2_loss": safe_mean(self.logs["q2_loss"]),
            "alpha_loss": safe_mean(self.logs["alpha_loss"]),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """
        Empties logs
        """
        self.logs = {}
        self.logs["q1_loss"] = []
        self.logs["q2_loss"] = []
        self.logs["policy_loss"] = []
        self.logs["alpha_loss"] = []
