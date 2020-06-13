import time
from collections import deque

from .vec_wrappers import VecEnvWrapper


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, history_length=0, info_keys=()):
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

    def reset(self):
        observation = self.venv.reset()
        self.episode_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_lens = np.zeros(self.n_envs, dtype=int)
        return observation

    def step(self, actions):
        observations, rewards, dones, infos = self.venv.step(actions)

        self.episode_returns += rewards
        self.episode_lens += 1

        new_infos = list(infos)
        for i in range(self.n_envs):
            if dones[i]:
                info = infos[i].copy()

                episode_info = {
                    "r": self.episode_returns[i],
                    "l": self.episode_lens[i],
                    "t": round(time.time() - self.tstart, 4),
                }

                for key in self.keys:
                    episode_info[key] = info[key]

                info["episode"] = episode_info

                if self.len:
                    self.returns_history.append(self.episode_returns[i])
                    self.lens_history.append(self.episode_lens[i])

                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lens[i] = 0

                new_infos[i] = info
        return observations, rewards, dones, new_infos
