from abc import abstractmethod

from genrl.environments.vec_env.vector_envs import VecEnv


class VecEnvWrapper(VecEnv):
    def __init__(self, venv):
        self.venv = venv
        super(VecEnvWrapper, self).__init__(envs=venv.envs, n_envs=venv.n_envs)

    def __getattr__(self, name):
        return getattr(self.venv, name)

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    def render(self, mode="human"):
        return self.venv.render(mode=mode)

    def close(self):
        self.venv.close()
