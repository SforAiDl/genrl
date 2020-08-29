import random
from collections import deque
from typing import NamedTuple, Tuple

import reverb
import numpy as np
import torch
import tensorflow as tf


class ReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class ReplayBuffer:
    def __init__(self, size, obs_shape, action_shape, n_envs=1):
        self.buffer_size = size
        self.n_envs = n_envs

        self.observations = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
            )
            + env.obs_shape,
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
            )
            + env.action_shape,
            dtype=np.float32,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_observations = np.zeros(
            (
                self.buffer_size,
                self.n_envs,
            )
            + env.obs_shape,
            dtype=np.float32,
        )
        self.pos = 0
        self._table = reverb.Table(
            name="uniform_experience_replay_buffer",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=1000,
            rate_limiter=reverb.rate_limiters.MinSize(2),
        )
        self._server = reverb.Server(tables=[self._table], port=None)
        self._client = reverb.Client(f"localhost:{self._server.port}")

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
        return [
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        ]

    def __len__(self) -> int:
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

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.alpha = alpha
        self.beta = beta
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
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(inp)
        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int, beta: float = None
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
        if beta is None:
            beta = self.beta

        total = len(self.buffer)
        priorities = np.asarray(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(total, batch_size, p=probabilities)

        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.asarray(weights, dtype=np.float32)

        samples = [self.buffer[i] for i in indices]
        (states, actions, rewards, next_states, dones) = map(np.stack, zip(*samples))

        return [
            torch.as_tensor(v, dtype=torch.float32)
            for v in [
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ]
        ]

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
            self.priorities[int(idx)] = priority.mean()

    def __len__(self) -> int:
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.buffer)

    @property
    def pos(self):
        return len(self.buffer)


class ReverbReplayBuffer:
    def __init__(
        self,
        size,
        batch_size,
        obs_shape,
        action_shape,
        action_dtype="discrete",
        reward_shape=(1,),
        done_shape=(1,),
        n_envs=1,
    ):
        self.size = size
        self.obs_shape = (n_envs, *obs_shape)
        self.action_shape = (n_envs, *action_shape)
        self.reward_shape = (n_envs, *reward_shape)
        self.done_shape = (n_envs, *done_shape)
        self.n_envs = n_envs
        self.action_dtype = np.int64 if action_dtype == "discrete" else np.float32

        self._pos = 0
        self._table = reverb.Table(
            name="replay_buffer",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=size,
            rate_limiter=reverb.rate_limiters.MinSize(2),
        )
        self._server = reverb.Server(tables=[self._table], port=None)
        self._server_address = f"localhost:{self._server.port}"
        self._client = reverb.Client(self._server_address)
        self._dataset = reverb.ReplayDataset(
            server_address=self._server_address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(np.float32, self.action_dtype, np.float32, np.float32, np.bool),
            shapes=(
                tf.TensorShape([n_envs, *obs_shape]),
                tf.TensorShape([n_envs, *action_shape]),
                tf.TensorShape([n_envs, *reward_shape]),
                tf.TensorShape([n_envs, *obs_shape]),
                tf.TensorShape([n_envs, *done_shape]),
            ),
        )
        self._iterator = self._dataset.batch(batch_size).as_numpy_iterator()

    def push(self, inp):
        i = []
        i.append(np.array(inp[0], dtype=np.float32).reshape(self.obs_shape))
        i.append(np.array(inp[1], dtype=self.action_dtype).reshape(self.action_shape))
        i.append(np.array(inp[2], dtype=np.float32).reshape(self.reward_shape))
        i.append(np.array(inp[3], dtype=np.float32).reshape(self.obs_shape))
        i.append(np.array(inp[4], dtype=np.bool).reshape(self.done_shape))

        self._client.insert(i, priorities={"replay_buffer": 1.0})
        if self._pos < self.size:
            self._pos += 1

    def extend(self, inp):
        for sample in inp:
            self.push(sample)

    def sample(self):
        sample = next(self._iterator)
        obs, a, r, next_obs, d = [torch.from_numpy(t) for t in sample.data]
        return obs, a, r, next_obs, d

    def __len__(self):
        return self._pos

    def __del__(self):
        self._server.stop()
