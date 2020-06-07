import torch
from collections import deque
import random
import numpy as np
from typing import Tuple
from .utils import get_obs_shape, get_action_dim


class ReplayBuffer:
    def __init__(self, size, env):
        self.obs_shape = get_obs_shape(env.observation_space)
        self.action_dim = get_action_dim(env.action_space)
        self.buffer_size = size
        self.n_envs = env.n_envs

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs,) + self.obs_shape, dtype=np.float32
        )
        self.pos = 0

    def push(self, inp):
        if self.pos >= self.buffer_size:
            self.observations = np.roll(self.observations, -1, axis=0)
            self.actions = np.roll(self.actions, -1, axis=0)
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.dones = np.roll(self.dones, -1, axis=0)
            self.next_observations = np.roll(self.next_observations, -1, axis=0)
            pos = self.buffer_size - 1
        else:
            pos = self.pos
        self.observations[pos] += np.array(inp[0]).copy()
        self.actions[pos] += np.array(inp[1]).copy()
        self.rewards[pos] += np.array(inp[2]).copy()
        self.next_observations[pos] += np.array(inp[3]).copy()
        self.dones[pos] += np.array(inp[4]).copy()
        self.pos += 1

    def sample(self, batch_size):
        if self.pos < self.buffer_size:
            indicies = np.random.randint(0, self.pos, size=batch_size)
        else:
            indicies = np.random.randint(0, self.buffer_size, size=batch_size)
        state = self.observations[indicies, :]
        action = self.actions[indicies, :]
        reward = self.rewards[indicies, :]
        next_state = self.next_observations[indicies, :]
        done = self.dones[indicies, :]
        return (
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        )

    def extend(self, inp):
        for sample in inp:
            if self.pos >= self.buffer_size:
                self.observations = np.roll(self.observations, -1, axis=0)
                self.actions = np.roll(self.actions, -1, axis=0)
                self.rewards = np.roll(self.rewards, -1, axis=0)
                self.dones = np.roll(self.dones, -1, axis=0)
                self.next_observations = np.roll(self.next_observations, -1, axis=0)
                pos = self.buffer_size - 1
            else:
                pos = self.pos
            self.observations[pos] = np.array(sample[0]).copy()
            self.actions[pos] = np.array(sample[1]).copy()
            self.rewards[pos] = np.array(sample[2]).copy()
            self.next_observations[pos] = np.array(sample[3]).copy()
            self.dones[pos] = np.array(sample[4]).copy()
            self.pos += 1


class PushReplayBuffer:
    """
    Implements the basic Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :type capacity: int
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, inp: Tuple) -> None:
        """
        Adds new experience to buffer

        :param inp: Tuple containing state, action, reward, next_state and done
        :type inp: tuple
        :returns: None
        """
        self.memory.append(inp)

    def extend(self, inp):
        self.memory.extend(inp)

    def sample(
        self, batch_size: int
    ) -> (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Returns randomly sampled experiences from replay memory

        :param batch_size: Number of samples per batch
        :type batch_size: int
        :returns: (Tuple composing of `state`, `action`, `reward`,
`next_state` and `done`)
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        print(state.shape)
        return (
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        )

    def get_len(self) -> int:
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return self.pos


class PrioritizedBuffer:
    """
    Implements the Prioritized Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :param alpha: Level of prioritization
    :type capacity: int
    :type alpha: int
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def push(self, inp: Tuple) -> None:
        """
        Adds new experience to buffer

        :param inp: (Tuple containing `state`, `action`, `reward`,
`next_state` and `done`)
        :type inp: tuple
        :returns: None
        """
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(inp)
        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> (
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """
        (Returns randomly sampled memories from replay memory along with their
respective indices and weights)

        :param batch_size: Number of samples per batch
        :param beta: (Bias exponent used to correct
Importance Sampling (IS) weights)
        :type batch_size: int
        :type beta: float
        :returns: (Tuple containing `states`, `actions`, `next_states`,
`rewards`, `dones`, `indices` and `weights`)
        """
        total = len(self.buffer)

        priorities = np.asarray(self.priorities)

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(total, batch_size, p=probabilities)

        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.asarray(weights, dtype=np.float32)

        samples = np.asarray(self.buffer, dtype=deque)[indices]
        (states, actions, rewards, next_states, dones) = map(np.stack, zip(*samples))

        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [states, actions, rewards, next_states, dones, indices, weights]
        )

    def update_priorities(self, batch_indices: Tuple, batch_priorities: Tuple) -> None:
        """
        Updates list of priorities with new order of priorities

        :param batch_indices: List of indices of batch
        :param batch_priorities: (List of priorities of the batch at the
specific indices)
        :type batch_indices: list or tuple
        :type batch_priorities: list or tuple
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = priority

    def get_len(self) -> int:
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.buffer)
