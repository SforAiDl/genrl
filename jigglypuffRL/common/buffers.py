import torch
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.memory = deque([], maxlen=size)

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
