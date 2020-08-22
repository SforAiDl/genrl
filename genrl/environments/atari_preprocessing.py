from typing import Tuple, Union

import cv2
import gym
import numpy as np
from gym.core import Wrapper
from gym.spaces import Box


class AtariPreprocessing(Wrapper):
    """
    Implementation for Image preprocessing for Gym Atari environments.
    Implements: 1) Frameskip 2) Grayscale 3) Downsampling to square image

    :param env: Atari environment
    :param frameskip: Number of steps between actions. \
E.g. frameskip=4 will mean 1 action will be taken for every 4 frames. It'll be\
 a tuple
if non-deterministic and a random number will be chosen from (2, 5)
    :param grayscale: Whether or not the output should be converted to \
grayscale
    :param screen_size: Size of the output screen (square output)
    :type env: Gym Environment
    :type frameskip: tuple or int
    :type grayscale: boolean
    :type screen_size: int
    """

    def __init__(
        self,
        env: gym.Env,
        frameskip: Union[Tuple, int] = (2, 5),
        grayscale: bool = True,
        screen_size: int = 84,
    ):
        super(AtariPreprocessing, self).__init__(env)

        self.frameskip = frameskip
        self.grayscale = grayscale
        self.screen_size = screen_size
        self.ale = (
            self.env.unwrapped.ale if hasattr(self.env.unwrapped, "ale") else None
        )

        if isinstance(frameskip, int):
            self.frameskip = (frameskip, frameskip + 1)

        # Redefine observation space for Atari environments
        if grayscale:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8
            )

        # Observation buffer to hold last two observations for max pooling
        self._obs_buffer = [
            np.empty(self.env.observation_space.shape[:2], dtype=np.uint8),
            np.empty(self.env.observation_space.shape[:2], dtype=np.uint8),
        ]

    # TODO(zeus3101) Add support for games with multiple lives

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Step through Atari environment for given action

        :param action: Action taken by agent
        :type action: NumPy array
        :returns: Current state, reward(for frameskip number of actions), \
done, info
        """
        frameskip = np.random.choice(range(*self.frameskip))
        index = 0

        reward = 0
        for timestep in range(frameskip):
            _, step_reward, done, info = self.env.step(action)
            reward += step_reward

            if done:
                break

            if timestep >= frameskip - 2:
                self._get_screen(index)
                index += 1

        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """
        Resets state of environment

        :returns: Initial state
        :rtype: NumPy array
        """
        self.env.reset()
        self._get_screen(0)
        self._obs_buffer[1].fill(0)

        return self._get_obs()

    def _get_screen(self, index: int) -> None:
        """
        Get the screen input given empty numpy array (from observation buffer)

        :param index: Index of the observation buffer that needs to be updated
        :type index: int
        """
        if self.grayscale:
            self.ale.getScreenGrayscale(self._obs_buffer[index])
        else:
            self.ale.getScreenRGB2(self._obs_buffer[index])

    def _get_obs(self) -> np.ndarray:
        """
        Performs max pooling on both states in observation buffer and \
resizes output to appropriate screen size.

        :returns: Output observation in required format
        :rtype: NumPy array
        """
        np.maximum(self._obs_buffer[0], self._obs_buffer[1], out=self._obs_buffer[0])

        obs = cv2.resize(
            self._obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        return np.array(obs, dtype=np.uint8)
