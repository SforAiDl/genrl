from collections import deque
import numpy as np

from gym.spaces import Box
from gym.core import Wrapper


class LazyFrames(object):
    """
    Efficient data structure to save each frame only once. \
Can use LZ4 compression to optimizer memory usage.
    """
    def __init__(self, frames, compress=False):
        if compress:
            from lz4.block import compress
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.compress = compress

    def __array__(self):
        if self.compress:
            from lz4.block import decompress
            frames = [
                np.frombuffer(
                    decompress(frame),
                    dtype=self._frames[0].dtype
                ).reshape(self._frames[0].shape)
                for frame in self._frames
            ]
        else:
            frames = self._frames
        
        return np.concatenate(frames, axis=-1)

    def __getitem__(self, index):
        return self.__array__()[index]

    def __len__(self):
        return len(self.__array__())

    def __eq__(self, other):
        return self.__array__() == other

    @property
    def shape(self):
        return self.__array__().shape


class FrameStack(Wrapper):
    """
    Wrapper to stack the last 4 observations of agent efficiently
    """
    def __init__(self, env, framestack=4, compress=False):
        super(FrameStack, self).__init__(env)

        self.env = env
        self._frames = deque([], maxlen=framestack)
        self.framestack = framestack

        low = np.repeat(
            np.expand_dims(self.env.observation_space.low, axis=0),
            framestack, axis=0
        )
        high = np.repeat(
            np.expand_dims(self.env.observation_space.high, axis=0),
            framestack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.env.observation_space.dtype
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._frames.append(observation)
        return self._get_state(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.framestack):
            self._frames.append(observation)
        return self._get_state()

    def _get_state(self):
        return np.array(LazyFrames(list(self._frames)))
