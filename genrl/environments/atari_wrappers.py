import numpy as np

import gym
from gym.core import Wrapper

from genrl.environments import GymWrapper, AtariPreprocessing, FrameStack


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
    def __init__(self):
        super(FireReset, self).__init__(env)
        self.env = env
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self):



DEFAULT_ATARI_WRAPPERS = [AtariPreprocessing, FrameStack]
ALL_ATARI_WRAPPERS = [AtariPreprocessing, NoopReset, FireReset, FrameStack]

def AtariEnv(
    env_id,
    wrapper_list=ALL_ATARI_WRAPPERS
):
    """
    Function to apply wrappers for all Atari envs by Trainer class

    :param env: Environment Name
    :param wrapper_list: List of wrappers to use on the environment
    :type env: string
    :type wrapper_list: list or tuple
    """
    gym_env = gym.make(env_id)

    if NoopReset in wrapper_list:
        assert 'NOOP' in gym_env.unwrapped.get_action_meanings()
    if FireReset in wrapper_list:
        assert 'FIRE' in gym_env.unwrapped.get_action_meanings()

    for wrapper in wrapper_list:
        gym_env = wrapper(gym_env)

    env = GymWrapper(gym_env)

    return env
