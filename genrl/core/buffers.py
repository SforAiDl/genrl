import random
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np
import torch


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
