import torch
import gym

from genrl.environments import GymWrapper, PreProcess

class NoFrameSkip(AtariWrapper):
    def __init__(self, env, n_envs=None):
        pass


class FrameSkip(AtariWrapper):
    def __init__(self, env, n_envs=None, frameskip=4):
        pass

class AtariWrapper(GymWrapper):
    def __init__(self, env, n_envs=None, frameskip=None):
        if frameskip is None:
            env = NoFrameSkip(env, n_envs)
        else:
            env = FrameSkip(env, n_envs, frameskip)
        
    def preprocess(self, env):
