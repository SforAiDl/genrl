from typing import Generator, NamedTuple, Optional, Union

import gym
import numpy as np
import torch

from genrl.environments.vec_env import VecEnv


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: (int) Max number of element in the buffer
    :param env: (Environment) The environment being trained on
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        env: Union[gym.Env, VecEnv],
        device: Union[torch.device, str] = "cpu",
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.env = env
        self.pos = 0
        self.full = False
        self.device = device

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            arr = arr.unsqueeze(-1)
            shape = shape + (1,)

        return arr.permute(1, 0, *(np.arange(2, len(shape)))).reshape(
            shape[0] * shape[1], *shape[2:]
        )

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(
        self,
        batch_size: int,
    ):
        """
        :param batch_size: (int) Number of element to sample
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ):
        """
        :param batch_inds: (torch.Tensor)
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array: (np.ndarray)
        :param copy: (bool) Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: (torch.Tensor)
        """
        if copy:
            return array.detach().clone()
        return array


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    :param buffer_size: (int) Max number of element in the buffer
    :param env: (Environment) The environment being trained on
    :param device: (torch.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        env: Union[gym.Env, VecEnv],
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, env, device)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros(
            *(self.buffer_size, self.env.n_envs, *self.env.obs_shape)
        )
        self.actions = torch.zeros(
            *(self.buffer_size, self.env.n_envs, *self.env.action_shape)
        )
        self.rewards = torch.zeros(self.buffer_size, self.env.n_envs)
        self.returns = torch.zeros(self.buffer_size, self.env.n_envs)
        self.dones = torch.zeros(self.buffer_size, self.env.n_envs)
        self.values = torch.zeros(self.buffer_size, self.env.n_envs)
        self.log_probs = torch.zeros(self.buffer_size, self.env.n_envs)
        self.advantages = torch.zeros(self.buffer_size, self.env.n_envs)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: torch.zeros,
        action: torch.zeros,
        reward: torch.zeros,
        done: torch.zeros,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: (torch.zeros) Observation
        :param action: (torch.zeros) Action
        :param reward: (torch.zeros)
        :param done: (torch.zeros) End of episode signal.
        :param value: (torch.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (torch.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = obs.detach().clone()
        self.actions[self.pos] = action.detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.dones[self.pos] = done.detach().clone()
        self.values[self.pos] = value.detach().clone().flatten()
        self.log_probs[self.pos] = log_prob.detach().clone().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.env.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.env.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.env.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
