import gym

from ..environments import (
    GymWrapper, AtariPreprocessing, FrameStack, NoopReset, FireReset
)

from typing import Union, List


def GymEnv(env_id: str) -> gym.Env:
    """
    Function to apply wrappers for all regular Gym envs by Trainer class

    :param env: Environment Name
    :type env: string
    """
    gym_env = gym.make(env_id)
    env = GymWrapper(gym_env)

    return env


def AtariEnv(env_id: str, wrapper_list: List = None, **kwargs) -> gym.Env:
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :param wrapper_list: List of wrappers to use on the environment
    :type env: string
    :type wrapper_list: list or tuple
    """
    DEFAULT_ATARI_WRAPPERS = [AtariPreprocessing, FrameStack]
    DEFAULT_ARGS = {
        "frameskip": (2, 5),
        "grayscale": True,
        "screen_size": 84,
        "max_noops": 25,
        "framestack": 4,
        "lz4_compress": False,
    }
    for key in DEFAULT_ARGS:
        if key not in kwargs:
            kwargs[key] = DEFAULT_ARGS[key]

    if wrapper_list is None:
        wrapper_list = DEFAULT_ATARI_WRAPPERS

    if "NoFrameskip" in env_id:
        kwargs["frameskip"] = (1, 2)
    elif "Deterministic" in env_id:
        kwargs["frameskip"] = (4, 5)

    env = gym.make(env_id)
    env = GymWrapper(env)

    if NoopReset in wrapper_list:
        assert "NOOP" in env.unwrapped.get_action_meanings()
    if FireReset in wrapper_list:
        assert "FIRE" in env.unwrapped.get_action_meanings()

    for wrapper in wrapper_list:
        if wrapper is AtariPreprocessing:
            env = wrapper(
                env, kwargs["frameskip"],
                kwargs["grayscale"], kwargs["screen_size"]
            )
        elif wrapper is NoopReset:
            env = wrapper(env, kwargs["max_noops"])
        elif wrapper is FrameStack:
            if kwargs["framestack"] > 1:
                env = wrapper(env, kwargs["framestack"], kwargs["lz4_compress"])
        else:
            env = wrapper(env)

    return env
