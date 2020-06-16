from typing import Dict, List

import gym

from ..environments import AtariPreprocessing, FrameStack, GymWrapper, NoopReset
from ..environments.vec_env import SerialVecEnv, SubProcessVecEnv, VecEnv, VecNormalize


def VectorEnv(
    env_id: str, n_envs: int = 2, parallel: int = False, env_type: str = "gym",
) -> VecEnv:
    """
    Chooses the kind of Vector Environment that is required

    :param env_id: Gym environment to be vectorised
    :param n_envs: Number of environments
    :param parallel: True if we want environments to run parallely and (
subprocesses, False if we want environments to run serially one after the other)
    :param env_type: Type of environment. Currently, we support ["gym", "atari"]
    :type env_id: string
    :type n_envs: int
    :type parallel: False
    :type env_type: string
    :returns: Vector Environment
    :rtype: VecEnv
    """
    envs = []
    wrapper = AtariEnv if env_type == "atari" else GymEnv

    envs = [wrapper(env_id) for _ in range(n_envs)]

    if parallel:
        venv = SubProcessVecEnv(envs, n_envs)
    else:
        venv = SerialVecEnv(envs, n_envs)

    venv = VecNormalize(venv)

    return venv


def GymEnv(env_id: str) -> gym.Env:
    """
    Function to apply wrappers for all regular Gym envs by Trainer class

    :param env: Environment Name
    :type env: string
    """
    gym_env = gym.make(env_id)
    env = GymWrapper(gym_env)
    return env


def AtariEnv(
    env_id: str, wrapper_list: List = [AtariPreprocessing, NoopReset, FrameStack]
) -> gym.Env:
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :type env: string
    :param wrapper_list: List of wrappers to use
    """
    env = gym.make(env_id)
    env = GymWrapper(env)

    if "NoFrameskip" in env_id:
        frameskip = 1
    elif "Deterministic" in env_id:
        frameskip = 4
    else:
        frameskip = (2, 5)

    for wrapper in wrapper_list:
        if wrapper is AtariPreprocessing:
            env = wrapper(env, frameskip)
        env = wrapper(env)

    return env
