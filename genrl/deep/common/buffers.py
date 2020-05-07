import torch
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Implements the basic Experience Replay Mechanism

    Args:
        :param size: (int) Size of the replay buffer
    """
    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)

    def push(self, x):
        """
        Adds new experience to buffer

        Args:
            :param x: (tuple) Tuple containing state, action, reward, \
next_state and done
        """
        self.memory.append(x)

    def sample(self, batch_size):
        """
        Returns randomly sampled experiences from replay memory
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
        """
        return len(self.memory)


class PrioritizedBuffer:
    """
    Implements the Prioritized Experience Replay Mechanism

    Args:
        :param capacity: (int) Size of the replay buffer
        :param prob_alpha: (int) Level of prioritization
    """
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, x):
        """
        Adds new experience to buffer and new priorities to priorities memory

        Args:
            :param x: (tuple) Tuple containing state, action, reward, \
next_state and done
        """
        state, action, reward, next_state, done = x

        assert state.ndim == next_state.ndim

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(x)
        else:
            self.buffer[self.pos] = x

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Returns randomly sampled memories from replay memory along with their
        respective indices and weights

        Args:
            :param batch_size: (int) Number of samples per batch
            :param beta: (int) Bias exponent used to correct \
Importance Sampling (IS) weights
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [states, actions, rewards, next_states, dones, indices, weights]
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Updates list of priorities with new order of priorities

        Args:
            :param batch_indices: (list or tuple) List of indices of batch
            :param batch_priorities: (list or tuple) List of priorities of \
the batch at the specific indices
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = prio

    def get_len(self):
        """
        Gives number of experiences in buffer currently
        """
        return len(self.buffer)
