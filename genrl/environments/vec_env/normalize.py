from typing import Any, Tuple

import numpy as np

from genrl.environments.vec_env.utils import RunningMeanStd
from genrl.environments.vec_env.vector_envs import VecEnv
from genrl.environments.vec_env.wrappers import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    Wrapper to implement Normalization of observations and rewards for VecEnvs

    :param venv: The Vectorized environment
    :param n_envs: Number of environments in VecEnv
    :param norm_obs: True if observations should be normalized, else False
    :param norm_reward: True if rewards should be normalized, else False
    :param clip_reward: Maximum absolute value for rewards
    :type venv: Vectorized Environment
    :type n_envs: int
    :type norm_obs: bool
    :type norm_reward: bool
    :type clip_reward: float
    """

    def __init__(
        self,
        venv: VecEnv,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_reward: float = 20.0,
    ):
        super(VecNormalize, self).__init__(venv)

        self.obs_rms = RunningMeanStd(shape=self.obs_shape) if norm_obs else False
        self.reward_rms = RunningMeanStd(shape=(1, 1)) if norm_reward else False

        self.clip_reward = clip_reward

    def __getattr__(self, name: str) -> Any:
        """
        Direct all other attribute calls to parent classes

        :param name: Attribute needed
        :type name: string
        :returns: Corresponding attribute of parent class
        """
        venv = super(VecNormalize, self).__getattribute__("venv")
        return getattr(venv, name)

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Steps through all the environments and normalizes the observations and rewards (if enabled)

        :param actions: Actions to be taken for the Vectorized Environment
        :type actions: Numpy Array
        :returns: States, rewards, dones, infos
        """
        states, rewards, dones, infos = self.venv.step(actions)

        states = self._normalize(self.obs_rms, None, states)
        rewards = self._normalize(self.reward_rms, self.clip_reward, rewards).reshape(
            self.n_envs,
        )

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
            batch = (batch - rms.mean) / np.sqrt(rms.var + 1e-8)
        if clip:
            batch = np.clip(batch, -clip, clip)
        return batch

    def reset(self) -> np.ndarray:
        """
        Resets Vectorized Environment

        :returns: Initial observations
        :rtype: Numpy Array
        """
        states = self.venv.reset()
        return self._normalize(self.obs_rms, None, states)

    def close(self):
        """
        Close all individual environments in the Vectorized Environment
        """
        self.venv.close()
