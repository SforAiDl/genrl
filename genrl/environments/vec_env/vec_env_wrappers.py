from .vector_envs import VecEnv


class VecWrapper(VecEnv):
    def __init__(self, envs, )