import numpy as np
from .vector_envs import VecEnv
from .utils import RunningMeanStd


class VecNormalize(VecEnv):
    def __init__(
        self,
        venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
    ):
        super(VecNormalize, self).__init__(venv)

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape) if norm_obs else False
        self.reward_rms = RunningMeanStd(shape=(1, 1)) if norm_reward else False

        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.rewards = np.zeros((self.n_envs,), dtype=np.float32)
        self.gamma = gamma

    def __getattr__(self, name):
        envs = super(VecNormalize, self).__getattribute__("envs")
        return getattr(envs, name)

    def step(self, actions):
        states, rewards, dones, infos = self.envs.step(actions)

        self.rewards = self.rewards * self.gamma + rewards
        states = self._normalize(self.obs_rms, self.clip_obs, states)
        rewards = self._normalize(self.reward_rms, self.clip_reward, rewards).reshape(self.n_envs,)

        self.rewards[dones.astype(bool)] = 0

        return states, rewards, dones, infos

    def _normalize(self, rms, clip, batch):
        if rms:
            rms.update(batch)
            batch = np.clip(
                (batch - rms.mean) / np.sqrt(rms.var + 1e-8), -clip, clip
            )
        return batch

    def reset(self):
        self.rewards = np.zeros(self.n_envs)
        states = self.envs.reset()
        return self._normalize(self.obs_rms, self.clip_obs, states)

    def close(self):
        self.envs.close()
