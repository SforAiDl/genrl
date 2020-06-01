import numpy as np
from gym.spaces import Box
from gym.core import ActionWrapper


class ClipAction(ActionWrapper):
    def __init__(self, env):
        super(ClipAction, self).__init__(env)
        assert isinstance(self.env.action_space, Box)

    @property
    def action(self, action):
        return np.clip(
            action, self.env.action_space.low, self.env.action_space.high
        )


class RescaleAction(ActionWrapper):
    def __init__(self, env, low, high):
        super(RescaleAction, self).__init__(env)
        assert isinstance(self.env.action_space, Box)
        assert high > low

        self.low = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            + low
        )
        self.high = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            + high
        )
        self.action_space = Box(
            low=low, high=high, shape=env.action_space.shape,
            dtype=env.action_space.dtype
        )

    @property
    def action(self, action):
        assert np.all(action >= self.low)
        assert np.all(action <= self.high)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high-low) * ((action-self.low) / (self.high-self.low))
        return np.clip(action, low, high)
