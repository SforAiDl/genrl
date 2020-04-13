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

class PrioritizedBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, x):
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
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (torch.as_tensor(v, dtype=torch.float32) for v in [states, actions, rewards, next_states, dones, indices, weights])
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = prio

    def get_len(self):
        return len(self.buffer)
