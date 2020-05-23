import gym
from gym.core import Wrapper

from genrl.environments import GymWrapper, AtariPreprocessing, FrameStack


class NoopReset(Wrapper):
    """
    Some Atari environments always reset to the same state. So we take \
a random number of some empty (noop) action to introduce some stochasticity.

    :param env: Atari environment
    :type env: Gym Environment
    """
    def __init__(self, env):
        super(NoopReset, self).__init__(env)
        self.env = env


class FireReset(Wrapper):
    """
    Some Atari environments do not actually do anything until a \
specific action (the fire action) is taken, so we make it take the \
action before starting the training process

    :param env: Atari environment
    :type env: Gym Environment
    """
    def __init__(self):
        super(FireReset, self).__init__(env)
        self.env = env


DEFAULT_ATARI_WRAPPERS = [AtariPreprocessing, FrameStack]

def AtariEnv(
    env_id,
    wrapper_list=DEFAULT_ATARI_WRAPPERS
):
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :type env: string
    """
    gym_env = gym.make(env_id)

    for wrapper in wrapper_list:
        gym_env = wrapper(gym_env)

    env = GymWrapper(gym_env)

    return env
