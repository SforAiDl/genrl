import numpy as np

import gym
from gym.core import Wrapper

from genrl.environments import (
    GymWrapper, AtariPreprocessing, FrameStack
)


class NoopReset(Wrapper):
    """
    Some Atari environments always reset to the same state. So we take \
a random number of some empty (noop) action to introduce some stochasticity.

    :param env: Atari environment
    :param max_noops: Maximum number of Noops to be taken
    :type env: Gym Environment
    :type max_noops: int
    """
    def __init__(self, env, max_noops=25):
        super(NoopReset, self).__init__(env)
        self.env = env
        self.max_noops = max_noops
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """
        Resets state of environment. Performs the noop action a \
random number of times to introduce stochasticity

        :returns: Initial state
        :rtype: NumPy array
        """
        self.env.reset()

        noops = np.random.randint(1, self.max_noops+1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    def step(self, action):
        """
        Step through underlying Atari environment for given action

        :param action: Action taken by agent
        :type action: NumPy array
        :returns: Current state, reward(for frameskip number of actions), \
done, info
        """
        return self.env.step(action)


class FireReset(Wrapper):
    """
    Some Atari environments do not actually do anything until a \
specific action (the fire action) is taken, so we make it take the \
action before starting the training process

    :param env: Atari environment
    :type env: Gym Environment
    """
    def __init__(self, env):
        super(FireReset, self).__init__(env)
        self.env = env

    def reset(self):
        """
        Resets state of environment. Performs the noop action a \
random number of times to introduce stochasticity

        :returns: Initial state
        :rtype: NumPy array
        """
        observation = self.env.reset()

        action_meanings = env.unwrapped.get_action_meanings()

        if action_meanings[1] == "FIRE" and len(action_meanings) >= 3:
            self.env.step(1)
            observation, _, _, _ = self.env.step(2)

        return observation


def AtariEnv(
    env_id,
    wrapper_list=None,
    **kwargs
):
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
        "lz4_compress": False
    }
    for key in DEFAULT_ARGS:
        if key not in kwargs:
            kwargs[key] = DEFAULT_ARGS[key]

    if wrapper_list is None:
        wrapper_list = DEFAULT_ATARI_WRAPPERS

    if "NoFrameskip" in env_id:
        kwargs['frameskip'] = 1
    elif "Deterministic" in env_id:
        kwargs['frameskip'] = 4

    env = gym.make(env_id)
    env = GymWrapper(env)

    if NoopReset in wrapper_list:
        assert 'NOOP' in env.unwrapped.get_action_meanings()
    if FireReset in wrapper_list:
        assert 'FIRE' in env.unwrapped.get_action_meanings()

    for wrapper in wrapper_list:
        if wrapper is AtariPreprocessing:
            env = wrapper(
                env, kwargs['frameskip'],
                kwargs['grayscale'], kwargs['screen_size']
            )
        elif wrapper is NoopReset:
            env = wrapper(
                env, kwargs['max_noops']
            )
        elif wrapper is FrameStack:
            if kwargs['framestack'] > 1:
                env = wrapper(
                    env, kwargs['framestack'], kwargs['lz4_compress']
                )
        else:
            env = wrapper(env)

    return env
