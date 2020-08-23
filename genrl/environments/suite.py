from typing import List

import gym

from genrl.environments import (
    AtariPreprocessing,
    FireReset,
    FrameStack,
    GymWrapper,
    NoopReset,
)
from genrl.environments.time_limit import AtariTimeLimit, TimeLimit
from genrl.environments.vec_env import SerialVecEnv, SubProcessVecEnv, VecEnv


from multiagent.environment import MultiAgentEnv
# from multiagent.scenarios.simple_spread import Scenario
import multiagent.scenarios as scenarios
import torch 
import numpy as np



def VectorEnv(
    env_id: str, n_envs: int = 2, parallel: int = False, env_type: str = "gym"
) -> VecEnv:
    """
    Chooses the kind of Vector Environment that is required

    :param env_id: Gym environment to be vectorised
    :param n_envs: Number of environments
    :param parallel: True if we want environments to run parallely and (
subprocesses, False if we want environments to run serially one after the other)
    :param env_type: Type of environment. Currently, we support ["gym", "atari", "multiagent"]
    :type env_id: string
    :type n_envs: int
    :type parallel: False
    :type env_type: string
    :returns: Vector Environment
    :rtype: object
    """
    if env_type == "atari":
        wrapper = AtariEnv
    elif env_type == "multi":
        wrapper = MultiagentEnv
    else:
        wrapper = GymEnv

    envs = [wrapper(env_id) for _ in range(n_envs)]

    if parallel:
        venv = SubProcessVecEnv(envs, n_envs)
    else:
        venv = SerialVecEnv(envs, n_envs)

    return venv


def GymEnv(env_id: str) -> gym.Env:
    """
    Function to apply wrappers for all regular Gym envs by Trainer class

    :param env: Environment Name
    :type env: string
    :returns: Gym Environment
    :rtype: object
    """
    env = gym.make(env_id)
    return GymWrapper(TimeLimit(env))


def AtariEnv(
    env_id: str,
    wrapper_list: List = [
        AtariPreprocessing,
        NoopReset,
        FireReset,
        AtariTimeLimit,
        FrameStack,
    ],
) -> gym.Env:
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :type env: string
    :param wrapper_list: List of wrappers to use
    :type wrapper_list: list or tuple
    :returns: Gym Atari Environment
    :rtype: object
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
        else:
            env = wrapper(env)

    return env


def MultiagentEnv(env_id: str)-> gym.Env:
    # load scenario from script
    scenario = scenarios.load(env_id + ".py").Scenario()
    benchmark = False
    # scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.isFinished)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.isFinished)

    return GymWrapper(env)
