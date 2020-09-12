import time
from collections import deque
from typing import Tuple

import numpy as np

from genrl.environments.vec_env.wrappers import VecEnv, VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    """
    Monitor class for VecEnvs. Saves important variables into the info dictionary

    :param venv: Vectorized Environment
    :param history_length: Length of history for episode rewards and episode lengths
    :param info_keys: Important variables to save
    :type venv: object
    :type history_length: int
    :type info_keys: tuple or list
    """

    def __init__(self, venv: VecEnv, history_length: int = 0, info_keys: Tuple = ()):
        super(VecMonitor, self).__init__(venv)

        self.len = history_length

        self.episode_returns = None
        self.episode_lens = None
        self.episode_count = 0
        self.tstart = time.time()

        self.keys = info_keys

        if self.len:
            self.returns_history = deque([], maxlen=self.len)
            self.lens_history = deque([], maxlen=self.len)

    def reset(self) -> np.ndarray:
        """
        Resets Vectorized Environment

        :returns: Initial observations
        :rtype: Numpy Array
        """
        observation = self.venv.reset()
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_lens = np.zeros(self.n_envs, dtype=int)
        return observation

    def step(self, actions: np.ndarray) -> Tuple:
        """
        Steps through all the environments and records important information

        :param actions: Actions to be taken for the Vectorized Environment
        :type actions: Numpy Array
        :returns: States, rewards, dones, infos
        """
        observations, rewards, dones, infos = self.venv.step(actions)

        self.episode_returns += rewards.numpy()
        self.episode_lens += 1

        new_infos = infos.copy()
        for i, done in enumerate(dones):
            if done:
                episode_info = {
                    "Episode Rewards": self.episode_returns[i],
                    "Episode Length": self.episode_lens[i],
                    "Time taken": round(time.time() - self.tstart, 4),
                }

                for key in self.keys:
                    episode_info[key] = new_infos[i][key]

                new_infos[i]["episode"] = episode_info

                if self.len:
                    self.returns_history.append(self.episode_returns[i])
                    self.lens_history.append(self.episode_lens[i])

                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lens[i] = 0
        return observations, rewards, dones, new_infos
