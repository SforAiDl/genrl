from typing import Any

import gym
import torch
from gym.core import Wrapper


class TorchWrapper(Wrapper):
    """
    Wraps an environment to deal with tensors avoiding conversion to numpy at frontend.

    Example:
        ``env = TorchWrapper(env)``

    Attributes:


    """

    def __init__(self, env: gym.Env, *args, **kwargs):
        super(TorchWrapper, self).__init__(env, *args, **kwargs)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        state, reward, done, info = self.env.step(action.numpy())
        state = torch.from_numpy(state)
        return state, reward, done, info

    def reset(self) -> torch.Tensor:
        return torch.as_tensor(self.env.reset())

    def sample(self) -> torch.Tensor:
        return torch.as_tensor(self.env.action_space.sample())

    @property
    def obs_shape(self):
        obs_shape = self.env.observation_space.shape
        return torch.as_tensor(obs_shape)

    @property
    def action_shape(self):
        action_shape = self.env.action_space.shape
        return torch.as_tensor(action_shape)

    def __getattr__(self, name: str) -> Any:
        """
        All other calls would go to base env
        """
        env = super(TorchWrapper, self).__getattribute__("env")
        return getattr(env, name)
