from collections import deque
from typing import List, Tuple

import gym
import numpy as np
from gym.core import Wrapper
from gym.spaces import Box


class LazyFrames(object):
    """
    Efficient data structure to save each frame only once. \
Can use LZ4 compression to optimizer memory usage.

    :param frames: List of frames that needs to converted \
to a LazyFrames data structure
    :param compress: True if we want to use LZ4 compression \
to conserve memory usage
    :type frames: collections.deque
    :type compress: boolean
    """

    def __init__(self, frames: List, compress: bool = False):
        if compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.compress = compress

    def __array__(self) -> np.ndarray:
        """
        Makes the LazyFrames object convertible to a NumPy array
        """
        if self.compress:
            from lz4.block import decompress

            frames = [
                np.frombuffer(decompress(frame), dtype=self._frames[0].dtype).reshape(
                    self._frames[0].shape
                )
                for frame in self._frames
            ]
        else:
            frames = self._frames

        return np.stack(frames, axis=0)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Return frame at index
        """
        return self.__array__()[index]

    def __len__(self) -> int:
        """
        Return length of data structure
        """
        return len(self.__array__())

    def __eq__(self, other: np.ndarray) -> bool:
        """
        Compares if data structure is equivalent to another object

        :param other: Other object for comparison
        :type other: object
        """
        return self.__array__() == other

    @property
    def shape(self) -> Tuple:
        """
        Returns dimensions of other object
        """
        return self.__array__().shape


class FrameStack(Wrapper):
    """
    Wrapper to stack the last few(4 by default) observations of \
agent efficiently

    :param env: Environment to be wrapped
    :param framestack: Number of frames to be stacked
    :param compress: True if we want to use LZ4 compression \
to conserve memory usage
    :type env: Gym Environment
    :type framestack: int
    :type compress: bool
    """

    def __init__(self, env: gym.Env, framestack: int = 4, compress: bool = True):
        super(FrameStack, self).__init__(env)

        self.env = env
        self._frames = deque([], maxlen=framestack)
        self.framestack = framestack

        low = np.repeat(
            np.expand_dims(self.env.observation_space.low, axis=0), framestack, axis=0
        )
        high = np.repeat(
            np.expand_dims(self.env.observation_space.high, axis=0), framestack, axis=0
        )

        self.observation_space = Box(
            low=low, high=high, dtype=self.env.observation_space.dtype
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Steps through environment

        :param action: Action taken by agent
        :type action: NumPy Array
        :returns: Next state, reward, done, info
        :rtype: NumPy Array, float, boolean, dict
        """
        observation, reward, done, info = self.env.step(action)
        self._frames.append(observation)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """
        Resets environment

        :returns: Initial state of environment
        :rtype: NumPy Array
        """
        observation = self.env.reset()
        for _ in range(self.framestack):
            self._frames.append(observation)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Gets observation given deque of frames

        :returns: Past few frames
        :rtype: NumPy Array
        """
        return np.array(LazyFrames(list(self._frames)))[np.newaxis, ...]
