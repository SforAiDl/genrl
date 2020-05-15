import torch
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Implements the basic Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :type capacity: int
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, x):
        """
        Adds new experience to buffer

        :param x: Tuple containing state, action, reward, next_state and done
        :type x: tuple
        :returns: None
        """
        self.memory.append(x)

    def sample(self, batch_size):
        """
        Returns randomly sampled experiences from replay memory

        :param batch_size: Number of samples per batch
        :type batch_size: int
        :returns: Tuple composing of `state`, `action`, `reward`, \
`next_state` and `done`
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [state, action, reward, next_state, done]
        )

    def get_len(self):
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.memory)


class PrioritizedBuffer:
    """
    Implements the Prioritized Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :param alpha: Level of prioritization
    :type capacity: int
    :type alpha: int
    """
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def push(self, x):
        """
        Adds new experience to buffer

        :param x: Tuple containing `state`, `action`, `reward`, \
`next_state` and `done`
        :type x: tuple
        :returns: None
        """
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(x)
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        """
        Returns randomly sampled memories from replay memory along with their
        respective indices and weights

        :param batch_size: Number of samples per batch
        :param beta: Bias exponent used to correct \
Importance Sampling (IS) weights
        :type batch_size: int
        :type beta: float
        :returns: Tuple containing `states`, `actions`, `next_states`, \
`rewards`, `dones`, `indices` and `weights`
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
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))

        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [states, actions, rewards, next_states, dones, indices, weights]
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Updates list of priorities with new order of priorities

        :param batch_indices: List of indices of batch
        :param batch_priorities: List of priorities of the batch at the \
specific indices
        :type batch_indices: list or tuple
        :type batch_priorities: list or tuple
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = priority

    def get_len(self):
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.buffer)
