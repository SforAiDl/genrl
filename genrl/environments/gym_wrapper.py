from typing import Any

import gym
import numpy as np

from ..deep.common.utils import get_obs_action_shape


class GymWrapper(gym.Wrapper):
    """
    Wrapper class for all Gym Environments

    :param env: Gym environment name
    :param n_envs: Number of environments. None if not vectorised
    :param parallel: If vectorised, should environments be run through \
serially or parallelly
    :type env: string
    :type n_envs: None, int
    :type parallel: boolean
    """

    def __init__(self, env: gym.Env):
        super(GymWrapper, self).__init__(env)
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.obs_shape, self.action_shape = get_obs_action_shape(env)

        self.state = None
        self.action = None
        self.reward = None
        self.done = False
        self.info = {}

    def __getattr__(self, name: str) -> Any:
        """
        All other calls would go to base env
        """
        env = super(GymWrapper, self).__getattribute__("env")
        return getattr(env, name)

    def _get_obs_action_shape(self):
        """
        Get the shapes of observation and action spaces
        """
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_shape = self.observation_space.shape
        else:
            raise NotImplementedError

        if isinstance(self.action_space, gym.spaces.Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (1,)
        else:
            raise NotImplementedError

        return obs_shape, action_shape

    def sample(self) -> np.ndarray:
        """
        Shortcut method to directly sample from environment's action space

        :returns: Random action from action space
        :rtype: NumPy Array
        """
        return self.env.action_space.sample()

    def render(self, mode: str = "human") -> None:
        """
        Renders all envs in a tiles format similar to baselines.

        :param mode: Can either be 'human' or 'rgb_array'. \
Displays tiled images in 'human' and returns tiled images in 'rgb_array'
        :type mode: string
        """
        self.env.render(mode=mode)

    def seed(self, seed: int = None) -> None:
        """
        Set environment seed

        :param seed: Value of seed
        :type seed: int
        """
        self.env.seed(seed)

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Steps the env through given action

        :param action: Action taken by agent
        :type action: NumPy array
        :returns: Next observation, reward, game status and debugging info
        """
        self.state, self.reward, self.done, self.info = self.env.step(action)
        self.action = action
        return self.state, self.reward, self.done, self.info

    def reset(self) -> np.ndarray:
        """
        Resets environment

        :returns: Initial state
        """
        return self.env.reset()

    def close(self) -> None:
        """
        Closes environment
        """
        self.env.close()
