import gym

from ..environments import GymWrapper, AtariPreprocessing, FrameStack, NoopReset
from ..environments.vec_env import VecEnv, SubProcessVecEnv, SerialVecEnv


def VectorEnv(
    env_id: str,
    n_envs: int = 2,
    parallel: bool = False,
    env_type: str = "gym",
    **kwargs
) -> VecEnv:
    """
    Chooses the kind of Vector Environment that is required

    :param env_id: Gym environment to be vectorised
    :param n_envs: Number of environments
    :param parallel: True if we want environments to run parallely and \
subprocesses, False if we want environments to run serially one after the other
    :param env_type: Type of environment. Currently, we support ['gym', 'atari']
    :type env_id: string
    :type n_envs: int
    :type parallel: False
    :type env_type: string
    :returns: Vector Environment
    :rtype: VecEnv
    """
    assert env_type in ["gym", "atari"]
    if env_type == "gym":
        Env = GymEnv
    elif env_type == "atari":
        Env = AtariEnv

    envs = []
    for _ in range(n_envs):
        env = Env(env_id, **kwargs)
        envs.append(env)

    if parallel:
        return SubProcessVecEnv(envs, n_envs)
    else:
        return SerialVecEnv(envs, n_envs)


def GymEnv(env_id: str) -> gym.Env:
    """
    Function to apply wrappers for all regular Gym envs by Trainer class

    :param env: Environment Name
    :type env: string
    """
    gym_env = gym.make(env_id)
    env = GymWrapper(gym_env)

    return env


def AtariEnv(env_id: str, **kwargs) -> gym.Env:
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :type env: string
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

    if "wrapper_list" not in kwargs.keys():
        wrapper_list = DEFAULT_ATARI_WRAPPERS
    else:
        wrapper_list = kwargs["wrapper_list"]

    if "NoFrameskip" in env_id:
        kwargs["frameskip"] = 1
    elif "Deterministic" in env_id:
        kwargs["frameskip"] = 4

    env = gym.make(env_id)
    env = GymWrapper(env)

    for wrapper in wrapper_list:
        if wrapper is AtariPreprocessing:
            env = wrapper(
                env, kwargs["frameskip"], kwargs["grayscale"], kwargs["screen_size"]
            )
        elif wrapper is NoopReset:
            env = wrapper(env, kwargs["max_noops"])
        elif wrapper is FrameStack:
            env = wrapper(env, kwargs["framestack"], kwargs["lz4_compress"])
        else:
            env = wrapper(env)
    return env
