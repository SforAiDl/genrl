import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_len=None):
        super(TimeLimit, self).__init__(env)

        if max_episode_len is None:
            max_episode_len = self.env.spec.max_episode_steps
        else:
            self.env.spec.max_episode_steps = max_episode_len

        self._max_episode_len = max_episode_len
        self._steps_taken = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._steps_taken += 1
        info["done"] = done
        if self._steps_taken >= self._max_episode_len:
            done = True
            info["done"] = False
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._steps_taken = 0
        return self.env.reset(**kwargs)


class AtariTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_len=None):
        super(AtariTimeLimit, self).__init__(env)

        if max_episode_len is None:
            self._max_episode_len = self.env.spec.max_episode_steps
        else:
            self.env.spec.max_episode_steps = max_episode_len
            self._max_episode_len = max_episode_len
        self._steps_taken = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._steps_taken += 1
        info["done"] = done
        if self._steps_taken >= self._max_episode_len:
            done = True
            info["done"] = False
        if done:
            if info["ale.lives"] != 0:
                done = False
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._steps_taken = 0
        return self.env.reset(**kwargs)
