import collections
from collections import OrderedDict

import numpy as np
import torch
from gym import GoalEnv, spaces

from genrl.environments.torch import TorchWrapper
from genrl.environments.vec_env import SubProcessVecEnv


class HERGoalEnvWrapper(GoalEnv):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.spaces = list(env.observation_space.spaces.values())

        if isinstance(self.spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces["achieved_goal"].shape
            self.obs_dim = env.observation_space.spaces["observation"].shape[0]
            self.goal_dim = goal_space_shape[0]

            if len(goal_space_shape) == 2:
                assert (
                    goal_space_shape[1] == 1
                ), "Only 1D observation spaces are supported yet"
            else:
                assert (
                    len(goal_space_shape) == 1
                ), "Only 1D observation spaces are supported yet"

        if isinstance(self.spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            self.observation_space = spaces.MultiBinary(total_dim)

        elif isinstance(self.spaces[0], spaces.Box):
            lows = np.concatenate([space.low for space in self.spaces])
            highs = np.concatenate([space.high for space in self.spaces])
            self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        elif isinstance(self.spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            self.observation_space = spaces.MultiDiscrete(dimensions)

        else:
            raise NotImplementedError(f"{type(self.spaces[0])} space is not supported")

        self.keys = ["observation", "achieved_goal", "desired_goal"]
        self.episode_reward = torch.zeros(1)

    def convert_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict[key] for key in self.keys])

    def convert_obs_to_dict(self, obs):
        return OrderedDict(
            [
                ("observation", obs[: self.obs_dim]),
                (
                    "achieved_goal",
                    obs[self.obs_dim : self.obs_dim + self.goal_dim],
                ),
                ("desired_goal", obs[self.obs_dim + self.goal_dim :]),
            ]
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward

        if "done" not in info.keys():
            info["done"] = done
        return self.convert_dict_to_obs(obs), reward, done, info

    def sample(self):
        return self.env.action_space.sample()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        self.episode_reward = torch.zeros(1)
        return self.convert_dict_to_obs(self.env.reset())

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
