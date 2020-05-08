import torch
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, x):
        self.memory.append(x)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [state, action, reward, next_state, done]
        )

    def get_len(self):
        return len(self.memory)


class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def push(self, x):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(x)
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        total = len(self.buffer)

        priorities = np.asarray(self.priorities)

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            total, batch_size, p=probabilities
        )

        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.asarray(weights, dtype=np.float32)

        samples = np.asarray(self.buffer, dtype=deque)[indices]
        states, actions, rewards, next_states, dones = map(
            np.stack, zip(*samples)
        )

        return (
            torch.as_tensor(v, dtype=torch.float32)
            for v in [
                states, actions, rewards, next_states, dones, indices, weights
            ]
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = priority

    def get_len(self):
        return len(self.buffer)
