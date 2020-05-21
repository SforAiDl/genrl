import numpy as np
import cv2

import torch
from torchvision import transforms

import gym
from gym.spaces import Box


class AtariPreprocessing(gym.Wrapper):
    def __init__(
        self, env, frameskip=4, grayscale=True, screen_size=84
    ):
        super(AtariPreprocessing, self)
        self.env = env
        self.frameskip = frameskip
        self.grayscale = grayscale
        self.screen_size = screen_size
        self.ale = env.unwrapped.ale

        if grayscale:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size),
                dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=(screen_size, screen_size, 3),
                dtype=np.uint8
            )

        self.screen_data = [
            np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            np.empty(env.observation_space.shape[:2], dtype=np.uint8)
        ]

    def step(self, action):
        total_reward = 0

        for timestep in range(self.frameskip):
            _, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break

            if timestep == self.frameskip - 2:
                self._get_screen(0)
            elif timestep == self.frameskip - 1:
                self._get_screen(1)
            
            return self._get_obs(), total_reward, done, info
    
    def reset(self, **kwargs):
        self.env.reset(kwargs)

        self._get_screen(0)
        self.screen_data[1].fill(0)

        return self._get_obs()

    def _get_screen(self, index):
        if self.grayscale:
            self.ale.getScreenGrayscale(self.screen_data[index])
        else:
            self.ale.getScreenRGB2(self.screen_data[index])

    def _get_obs(self):
        if self.frameskip > 1:
            np.maximum(
                self.screen_data[0],
                self.screen_data[1],
                out=self.screen_data[0]
            )

        obs = cv2.resize(
            self.screen_data[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA
        )

        return np.array(obs, dtype=np.uint8)
        