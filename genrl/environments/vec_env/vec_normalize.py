from typing import Any, Tuple

import numpy as np

from .utils import RunningMeanStd
from .vector_envs import VecEnv


class VecNormalize(VecEnv):
    """
    Wrapper to implement Normalization of observations and rewards for VecEnvs

    :param venv: The Vectorized environment
    :param norm_obs: True if observations should be normalized, else False
    :param norm_reward: True if rewards should be normalized, else False
    :param clip_obs: Maximum absolute value for observations
    :param clip_reward: Maximum absolute value for rewards
    :param gamma: Discount Factor used in calculation of returns
    :type venv: Vectorized Environment
    :type norm_obs: bool
    :type norm_reward: bool
    :type clip_obs: float
    :type clip_reward: float
    :type gamma: float
    """

    def __init__(
        self,
        venv: VecEnv,
        norm_obs: bool = True,
        norm_reward: bool = False,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
    ):
        super(VecNormalize, self).__init__(venv)

        self.obs_rms = (
            RunningMeanStd(shape=self.observation_space.shape) if norm_obs else False
        )
        self.reward_rms = RunningMeanStd(shape=(1, 1)) if norm_reward else False

        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.returns = np.zeros((self.n_envs,), dtype=np.float32)
        self.gamma = gamma

    def __getattr__(self, name: str) -> Any:
        """
        Direct all other attribute calls to parent classes

        :param name: Attribute needed
        :type name: string
        :returns: Corresponding attribute of parent class
        """
        envs = super(VecNormalize, self).__getattribute__("envs")
        return getattr(envs, name)

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Steps through all the environments and normalizes the observations and rewards (if enabled)

        :param actions: Actions to be taken for the Vectorized Environment
        :type actions: Numpy Array
        :returns: States, rewards, dones, infos
        """
        states, rewards, dones, infos = self.envs.step(actions)

        self.returns = self.returns * self.gamma + rewards
        states = self._normalize(self.obs_rms, self.clip_obs, states)
        rewards = self._normalize(self.reward_rms, self.clip_reward, rewards).reshape(
            self.n_envs,
        )

        self.returns[dones.astype(bool)] = 0

        return states, rewards, dones, infos

    def _normalize(
        self, rms: RunningMeanStd, clip: float, batch: np.ndarray
    ) -> np.ndarray:
        """
        Function to normalize and clip a given RMS

        :param rms: Running mean standard deviation object to calculate new mean and new variance
        :param clip: Maximum Absolute value of observation/reward
        :param batch: Batch of observations/rewards to be normalized and clipped
        :type rms: object
        :type clip: float
        :type batch: Numpy Array
        :returns: Normalized observations/rewards
        :rtype: Numpy Array
        """
        if rms:
            rms.update(batch)
            batch = np.clip((batch - rms.mean) / np.sqrt(rms.var + 1e-8), -clip, clip)
        return batch

    def reset(self) -> np.ndarray:
        """
        Resets Vectorized Environment

        :returns: Initial observations
        :rtype: Numpy Array
        """
        self.returns = np.zeros(self.n_envs)
        states = self.envs.reset()
        return self._normalize(self.obs_rms, self.clip_obs, states)

    def close(self):
        """
        Close all individual environments in the Vectorized Environment
        """
        self.envs.close()
