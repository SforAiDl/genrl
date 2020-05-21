import gym
import torch

from genrl.environments import BaseWrapper
from genrl.deep.common import venv


class GymWrapper(BaseWrapper):
    """
    Wrapper class for all Gym Environments

    :param env: Gym environment name
    :param n_envs: Number of environments. None if not vectorised
    :type env: string
    :type n_envs: None, int
    """
    def __init__(self, env, n_envs=None):
        if n_envs is None:
            self._vec = False
            self.env = gym.make(env)
        else:
            self._vec = True
            self.env = venv(env, n_envs)

    def __getattr__(self, name):
        """
        All other calls would go to base env
        """
        env = super(GymWrapper, self).__getattribute__('env')
        return getattr(env, name)

    def observation_space(self):
        """
        Returns observation space of environment
        """
        if self._vec:
            raise NotImplementedError
        else:
            return self.env.observation_space

    def action_space(self):
        """
        Return action space of environment
        """
        if self._vec:
            raise NotImplementedError
        else:
            return self.env.action_space

    #TODO(zeus3101) Get get_state, set_state, get_info, get_done methods

    def render(self, mode="human"):
        """
        Renders all envs in a tiles format similar to baselines.

        :param mode: Can either be 'human' or 'rgb_array'. \
Displays tiled images in 'human' and returns tiled images in 'rgb_array'
        :type mode: string
        """
        self.env.render(mode=mode)

    def seed(self, seed):
        """
        Set environment seed

        :param seed: Value of seed
        :type seed: int
        """
        self.env.seed(seed)

    def step(self, action):
        """
        Steps the env through given action

        :param action: Action taken by agent
        :type action: NumPy array
        """
        self.env.step(action)

    def reset(self):
        """
        Resets environment
        """
        self.env.reset()
    
    def close(self):
        """
        Closes environment
        """
        self.env.close()
