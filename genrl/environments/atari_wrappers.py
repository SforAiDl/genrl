import torch
import gym

from genrl.environments import GymWrapper, AtariPreprocessing


class FrameStack(GymWrapper):
    def __init__(self):
        pass


class NoopReset(GymWrapper):
    def __init__(self):
        pass


class FireReset(GymWrapper):
    def __init__(self):
        pass


class Atari(GymWrapper):
    """
    General Atari Wrapper to use for all Atari envs by Trainer class

    :param env: Environment Name
    :type env: string
    """
    def __init__(self, env):
        super(Atari, self).__init__(env)
