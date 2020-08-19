import gc
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from a2c import *
from torch.autograd import Variable
from torch.distributions import Categorical

from genrl.deep.agents.base import OnPolicyAgent
from genrl.deep.common import (
    BaseActorCritic,
    RolloutBuffer,
    get_env_properties,
    get_model,
    safe_mean,
)
from genrl.environments.vec_env import VecEnv


class A2C(OnPolicyAgent):
    """
    Advantage Actor Critic algorithm (A2C)
    The synchronous version of A3C
    Paper: https://arxiv.org/abs/1602.01783

    :param network: The deep neural network
    :param env: The environment to learn from
    :param gamma: Discount factor
    :param actor_batch_size: Update batch size
    :param lr_policy: Policy Network learning rate
    :param lr_value: Value Network learning rate
    :param num_episodes: Number of episodes
    :param timesteps_per_actorbatch: Number of timesteps per epoch
    :param max_ep_len: Maximum timesteps in an episode
    :param layers: Number of neurons in hidden layers
    :param noise: Noise function to use
    :param noise_std: Standard deviation for action noise
    :param seed: Seed for reproducing results
    :param render: True if environment is to be rendered, else False
    :param device: Device to use for Tensor operation ['cpu', 'cuda']
    :param rollout_size: Rollout Buffer Size
    :param val_coeff: Coefficient of value loss in overall loss function
    :param entropy_coeff: Coefficient of entropy loss in overall loss function
    :type network: string or BaseActorCritic
    :type env: Gym Environment
    :type gamma: float
    :type actor_batch_size: int
    :type lr_policy: float
    :type lr_value: float
    :type num_episodes: int
    :type timesteps_per_actorbatch: int
    :type max_ep_len: int
    :type layers: tuple or list
    :type noise: function
    :type noise_std: float
    :type seed: int
    :type render: boolean
    :type device: string
    :type rollout_size: int
    :type val_coeff: float
    :type entropy_coeff: float
    """

    def __init__(
        self,
        network: Union[str, BaseActorCritic],
        env: Union[gym.Env, VecEnv],
        batch_size: int = 256,
        gamma: float = 0.99,
        lr_policy: float = 0.01,
        lr_value: float = 0.1,
        layers: Tuple = (32, 32),
        rollout_size: int = 2048,
        noise: Any = None,
        noise_std: float = 0.1,
        **kwargs
    ):
        super(A2C, self).__init__(
            network,
            env,
            batch_size=batch_size,
            layers=layers,
            gamma=gamma,
            lr_policy=lr_policy,
            lr_value=lr_value,
            rollout_size=rollout_size,
            **kwargs
        )

        self.noise = noise
        self.noise_std = noise_std
        self.value_coeff = kwargs.get("value_coeff", 0.5)
        self.entropy_coeff = kwargs.get("entropy_coeff", 0.01)

        self.buffer_class = kwargs.get("buffer_class", RolloutBuffer)

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self) -> None:
        """
        Creates actor critic model and initialises optimizers
        """
        if isinstance(self.network, str):
            input_dim, action_dim, discrete, action_lim = get_env_properties(
                self.env, self.network
            )
            self.ac = get_model("ac", self.network)(
                input_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
            ).to(self.device)

        else:
            self.ac = self.network.to(self.device)
            action_dim = self.network.action_dim

        if self.noise is not None:
            self.noise = self.noise(
                np.zeros_like(action_dim), self.noise_std * np.ones_like(action_dim)
            )

        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

        self.rollout = self.buffer_class(self.rollout_size, self.env)

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
        state = torch.as_tensor(state).float().to(self.device)

        # create distribution based on actor output
        action, dist = self.ac.get_action(state, deterministic=False)
        value = self.ac.get_value(state)

        return action.detach().cpu().numpy(), value, dist.log_prob(action).cpu()

    def get_traj_loss(self, values, dones) -> None:
        """
        (Get trajectory of agent to calculate discounted rewards and
calculate losses)
        """
        self.rollout.compute_returns_and_advantage(values.detach().cpu().numpy(), dones)

    def get_value_log_probs(self, state, action):
        state, action = state.to(self.device), action.to(self.device)
        _, dist = self.ac.get_action(state, deterministic=False)
        value = self.ac.get_value(state)
        return value, dist.log_prob(action).cpu()

    def update_policy(self) -> None:
        for rollout in self.rollout.get(self.batch_size):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.advantages * log_prob
            policy_loss = -torch.mean(policy_loss)
            self.logs["policy_loss"].append(policy_loss.item())

            value_loss = self.value_coeff * F.mse_loss(rollout.returns, values.cpu())
            self.logs["value_loss"].append(torch.mean(value_loss).item())

            entropy_loss = (torch.exp(log_prob) * log_prob).sum()
            self.logs["policy_entropy"].append(entropy_loss.item())

            actor_loss = policy_loss + self.entropy_coeff * entropy_loss

            self.optimizer_policy.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
            self.optimizer_value.step()

    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Loads important hyperparameters that need to be loaded or saved

        :returns: Hyperparameters that need to be saved or loaded
        :rtype: dict
        """
        hyperparams = {
            "network": self.network,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "rollout_size": self.rollout_size,
            "policy_weights": self.ac.actor.state_dict(),
            "value_weights": self.ac.critic.state_dict(),
        }
        return hyperparams

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.actor.load_state_dict(weights["policy_weights"])
        self.ac.critic.load_state_dict(weights["value_weights"])

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
        self.logs = {}
        self.logs["policy_loss"] = []
        self.logs["value_loss"] = []
        self.logs["policy_entropy"] = []
        self.rewards = []


class A2CAgent:
    def __init__(self, env, lr=2e-4, gamma=0.99, load_model=None, entropy_weight=0.008):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.w = entropy_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_agents = self.env.n

        self.input_dim = self.env.observation_space[0].shape[0]
        self.action_dim = self.env.action_space[0].n

        self.actorcritic = None
        self.optimizer = None

        self.episode_reward = 0
        self.steps_per_episode = None
        self.final_step = 0

        self.values = None
        self.logits = None
        self.states = None
        self.next_states = None
        self.actions = None
        self.dones = None
        self.rewards = None

        self.value_loss = None
        self.policy_loss = None
        self.entropy = None
        self.total_loss = None

        self.grad_norm = None

        self.setup_model(load_model)

    def setup_model(self, load_model):

        self.actorcritic = CentralizedActorCritic(self.input_dim, self.action_dim).to(
            self.device
        )
        if load_model is not None:
            self.actorcritic.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

        self.optimizer = optim.Adam(self.actorcritic.parameters(), lr=self.lr)

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.actorcritic.get_action(states[i], self.device, False)
            actions.append(action)
        return actions

    def collect_rollouts(self, states):
        trajectory = []
        self.episode_reward = 0
        self.final_step = 0
        current_states = states

        for step in range(self.steps_per_episode):

            actions = self.get_actions(current_states)
            next_states, rewards, dones, info = self.env.step(actions)
            self.episode_reward += np.sum(rewards)

            if all(dones) or step == self.steps_per_episode - 1:

                dones = [1 for _ in range(self.num_agents)]
                trajectory.append([states, next_states, actions, rewards, dones])
                print("REWARD: {} \n".format(np.round(self.episode_reward, decimals=4)))
                print("*" * 100)
                self.final_step = step
                break
            else:
                dones = [0 for _ in range(self.num_agents)]
                trajectory.append([states, next_states, actions, rewards, dones])
                current_states = next_states
                self.final_step = step

        self.states = torch.FloatTensor([sars[0] for sars in trajectory]).to(
            self.device
        )
        self.next_states = torch.FloatTensor([sars[1] for sars in trajectory]).to(
            self.device
        )
        self.actions = torch.LongTensor([sars[2] for sars in trajectory]).to(
            self.device
        )
        self.rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(
            self.device
        )
        self.dones = torch.LongTensor([sars[4] for sars in trajectory])

        self.logits, self.values = self.actorcritic.forward(self.states)

        return self.values, self.dones

    def get_traj_loss(self, curr_Q, done):
        discounted_rewards = np.asarray(
            [
                [
                    torch.sum(
                        torch.FloatTensor(
                            [
                                self.gamma ** i
                                for i in range(self.rewards[k][j:].size(0))
                            ]
                        )
                        * self.rewards[k][j:]
                    )
                    for j in range(self.rewards.size(0))
                ]
                for k in range(self.num_agents)
            ]
        )
        discounted_rewards = np.transpose(discounted_rewards)
        value_targets = self.rewards + torch.FloatTensor(discounted_rewards).to(
            self.device
        )
        value_targets = value_targets.unsqueeze(dim=-1)
        self.value_loss = F.smooth_l1_loss(curr_Q, value_targets)

        dists = F.softmax(self.logits, dim=-1)
        probs = Categorical(dists)

        self.entropy = -torch.mean(
            torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=-1)
        )

        advantage = value_targets - curr_Q
        self.policy_loss = -probs.log_prob(self.actions) * advantage.detach()
        self.policy_loss = self.policy_loss.mean()

        self.total_loss = self.policy_loss + self.value_loss - self.w * self.entropy

    def update_policy(self):
        self.optimizer.zero_grad()
        self.total_loss.backward(retain_graph=False)
        self.grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actorcritic.parameters(), 0.5
        )
        self.optimizer.step()

    def get_logging_params(self):
        logging_params = {
            "Loss/Entropy loss": self.entropy.item(),
            "Loss/Value Loss": self.value_loss.item(),
            "Loss/Policy Loss": self.policy_loss,
            "Loss/Total Loss": self.total_loss,
            "Gradient Normalization/Grad Norm": self.grad_norm,
            "Reward Incurred/Length of the episode": self.final_step,
            "Reward Incurred/Reward": self.episode_reward,
        }
        return logging_params

    def get_hyperparams(self):
        hyperparams = {
            "gamma": self.gamma,
            "entropy_weight": self.w,
            "lr_actor": self.lr,
            "lr_critic": self.lr,
            "actorcritic_weights": self.actorcritic.state_dict(),
        }

        return hyperparams
