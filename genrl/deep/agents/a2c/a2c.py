from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

from ....environments.vec_env import VecEnv
from ...common import (
    RolloutBuffer,
    get_model,
    load_params,
    safe_mean,
    save_params,
    set_seeds,
)


class A2C:
    """
    Advantage Actor Critic algorithm (A2C)
    The synchronous version of A3C
    Paper: https://arxiv.org/abs/1602.01783

    :param network_type: The deep neural network layer types ['mlp']
    :param env: The environment to learn from
    :param gamma: Discount factor
    :param actor_batch_size: Update batch size
    :param lr_actor: Policy Network learning rate
    :param lr_critic: Value Network learning rate
    :param num_episodes: Number of episodes
    :param timesteps_per_actorbatch: Number of timesteps per epoch
    :param max_ep_len: Maximum timesteps in an episode
    :param layers: Number of neurons in hidden layers
    :param noise: Noise function to use
    :param noise_std: Standard deviation for action noise
    :param seed: Seed for reproducing results
    :param render: True if environment is to be rendered, else False
    :param device: Device to use for Tensor operation ['cpu', 'cuda']
    :param run_num: Model run number if it has already been trained
    :param save_model: Directory the user wants to save models to
    :param save_interval: Number of steps between saves of models
    :param rollout_size: Rollout Buffer Size
    :param val_coeff: Coefficient of value loss in overall loss function
    :param entropy_coeff: Coefficient of entropy loss in overall loss function
    :type network_type: string
    :type env: Gym Environment
    :type gamma: float
    :type actor_batch_size: int
    :type lr_a: float
    :type lr_c: float
    :type num_episodes: int
    :type timesteps_per_actorbatch: int
    :type max_ep_len: int
    :type layers: tuple or list
    :type noise: function
    :type noise_std: float
    :type seed: int
    :type render: boolean
    :type device: string
    :type run_num: int
    :type save_model: string
    :type save_interval: int
    :type rollout_size: int
    :type val_coeff: float
    :type entropy_coeff: float
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, VecEnv],
        gamma: float = 0.99,
        actor_batch_size: int = 64,
        lr_actor: float = 0.01,
        lr_critic: float = 0.1,
        num_episodes: int = 100,
        timesteps_per_actorbatch: int = 4000,
        max_ep_len: int = 1000,
        layers: Tuple = (32, 32),
        noise: Any = None,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        save_interval: int = 1000,
        rollout_size: int = 2048,
        val_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
    ):
        self.network_type = network_type
        self.env = env
        self.gamma = gamma
        self.actor_batch_size = actor_batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.max_ep_len = max_ep_len
        self.layers = layers
        self.noise = noise
        self.noise_std = noise_std
        self.seed = seed
        self.render = render
        self.run_num = run_num
        self.save_interval = save_interval
        self.save_model = None
        self.save = save_params
        self.load = load_params
        self.rollout_size = rollout_size
        self.val_coeff = val_coeff
        self.entropy_coeff = entropy_coeff

        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["policy_entropy"] = []
        self.rewards = []

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        self.create_model()

    def create_model(self) -> None:
        """
        Creates actor critic model and initialises optimizers
        """
        (state_dim, action_dim, discrete, action_lim) = get_env_properties(
            self.env, self.network_type
        )

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        self.actor_optimizer = opt.Adam(self.ac.actor.parameters(), lr=self.lr_actor)

        self.critic_optimizer = opt.Adam(self.ac.critic.parameters(), lr=self.lr_critic)

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

        # create distribution based on actor output
        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state).unsqueeze(0)

        return a, val, c.log_prob(a)

    def get_traj_loss(self, value: np.ndarray, done: bool) -> None:
        """
        (Get trajectory of agent to calculate discounted rewards and
calculate losses)

        :param value: Value of a state
        :param done: True if the state is terminal, else False
        :type value: NumPy Array
        :type done: boolean
        """
        self.rollout.compute_returns_and_advantage(value.detach().cpu().numpy(), done)

    def get_value_log_probs(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate the value of a given state and the log probability of taking a given action in the given state

        :param state: State value
        :param action: Action to be taken
        :type state: NumPy Array
        :type action: NumPy Array
        :returns: Value and Log probability of Action
        :rtype: NumPy Array, NumPy Array
        """
        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state)
        return val, c.log_prob(action)

    def update_policy(self) -> None:
        """
        Function to calculate loss from rollouts and update the policy
        """
        for rollout in self.rollout.get(256):

            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.advantages * log_prob
            policy_loss = -torch.mean(policy_loss)
            self.logs["policy_loss"].append(policy_loss.item())

            value_loss = self.val_coeff * F.mse_loss(rollout.returns, values)
            self.logs["value_loss"].append(torch.mean(value_loss).item())

            entropy_loss = (torch.exp(log_prob) * log_prob).sum()
            self.logs["policy_entropy"].append(entropy_loss.item())

            actor_loss = policy_loss + self.entropy_coeff * entropy_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
            self.critic_optimizer.step()

    def collect_rollouts(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Function to calculate rollouts

        :param state: Initial state before calculating rollouts
        :type state: NumPy Array
        """
        for i in range(2048):
            # with torch.no_grad():
            action, values, old_log_probs = self.select_action(state)

            next_state, reward, done, _ = self.env.step(np.array(action))
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                done,
                values.detach(),
                old_log_probs.detach(),
            )

            state = next_state

            for i, di in enumerate(done):
                if di:
                    self.rewards.append(self.epoch_reward[i])
                    self.epoch_reward[i] = 0

        return values, done

    def learn(self):  # pragma: no cover
        """
        Trains actor critic model
        """
        for episode in range(self.num_episodes):
            episode_reward = 0

            self.update(episode)

            if episode % 5 == 0:
                print("Episode: {}, Reward: {}".format(episode, episode_reward))

        self.env.close()

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
            "actor_batch_size": self.actor_batch_size,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "actor_weights": self.ac.actor.state_dict(),
            "critic_weights": self.ac.critic.state_dict(),
        }

        return hyperparams

    def get_logging_params(self) -> Dict[str, Any]:
        """
        :returns: Logging parameters for monitoring training
        :rtype: dict
        """

        logs = {
            "policy_loss": safe_mean(self.logs["policy_loss"]),
            "value_loss": safe_mean(self.logs["value_loss"]),
            "policy_entropy": safe_mean(self.logs["policy_entropy"]),
            "mean_reward": safe_mean(self.rewards),
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
    algo = A2C("mlp", env)
    algo.learn()
