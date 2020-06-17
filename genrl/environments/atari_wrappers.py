import gym
import numpy as np
from gym.core import Wrapper


class NoopReset(Wrapper):
    """
    Some Atari environments always reset to the same state. So we take \
a random number of some empty (noop) action to introduce some stochasticity.

    :param env: Atari environment
    :param max_noops: Maximum number of Noops to be taken
    :type env: Gym Environment
    :type max_noops: int
    """

    def __init__(self, env: gym.Env, max_noops: int = 30):
        super(NoopReset, self).__init__(env)
        self.max_noops = max_noops
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self) -> np.ndarray:
        """
        Resets state of environment. Performs the noop action a \
random number of times to introduce stochasticity

        :returns: Initial state
        :rtype: NumPy array
        """
        self.env.reset()

        noops = np.random.randint(1, self.max_noops + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    def step(self, action: np.ndarray) -> np.ndarray:
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

    def __init__(self, env: gym.Env):
        super(FireReset, self).__init__(env)

    def reset(self) -> np.ndarray:
        """
        Resets state of environment. Performs the noop action a \
random number of times to introduce stochasticity

        :returns: Initial state
        :rtype: NumPy array
        """
        observation = self.env.reset()

        action_meanings = self.env.unwrapped.get_action_meanings()

        if action_meanings[1] == "FIRE" and len(action_meanings) >= 3:
            self.env.step(1)
            observation, _, _, _ = self.env.step(2)

        return observation
