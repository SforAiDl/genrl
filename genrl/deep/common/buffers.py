import gym
import torch
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size, env):
        self.size = size
        # self.memory = deque([], maxlen=size)
        self.observations = np.zeros((size, env.n_envs, env.observation_space.shape[0]))
        if isinstance(env.envs[0].unwrapped, gym.envs.classic_control.CartPoleEnv):
            self.actions = np.zeros((size, env.n_envs, 1))
        else:
            self.actions = np.zeros((size, env.n_envs, env.action_space.shape[0]))
        self.rewards = np.zeros((size, env.n_envs))
        self.dones = np.zeros((size, env.n_envs))
        self.next_observations = np.zeros((size, env.n_envs, env.observation_space.shape[0]))
        self.pos = 0

    def push(self, x):
        if self.pos >= self.size:
            self.observations = np.roll(self.observations, -1, axis=0)
            self.actions = np.roll(self.actions, -1, axis=0)
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.dones = np.roll(self.dones, -1, axis=0)
            self.next_observations = np.roll(self.next_observations, -1, axis=0)
            pos = self.size-1
        else:
            pos = self.pos
        self.observations[pos] += np.array(x[0]).copy()
        self.actions[pos] += np.array(x[1]).copy()
        self.rewards[pos] += np.array(x[2]).copy()
        self.next_observations[pos] += np.array(x[3]).copy()
        self.dones[pos] += np.array(x[4]).copy()
        self.pos += 1

    def sample(self, batch_size):
        if self.pos < self.size:
            indicies = np.random.randint(0, self.pos, size=batch_size)
        else:
            indicies = np.random.randint(0, self.size, size=batch_size)
        state = self.observations[indicies,:]
        action = self.actions[indicies,:]
        reward = self.rewards[indicies,:]
        next_state = self.next_observations[indicies,:]
        done = self.dones[indicies,:]
        # print(state.shape)
        # batch = random.sample(self.memory, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        )

    def get_len(self):
        return self.pos

    def extend(self, x):
        for sample in x:
            if self.pos >= self.size:
                self.observations = np.roll(self.observations, -1, axis=0)
                self.actions = np.roll(self.actions, -1, axis=0)
                self.rewards = np.roll(self.rewards, -1, axis=0)
                self.dones = np.roll(self.dones, -1, axis=0)
                self.next_observations = np.roll(self.next_observations, -1, axis=0)
                pos = self.size-1
            else:
                pos = self.pos
            self.observations[pos] += np.array(sample[0]).copy()
            self.actions[pos] += np.array(sample[1]).copy()
            self.rewards[pos] += np.array(sample[2]).copy()
            self.next_observations[pos] += np.array(sample[3]).copy()
            self.dones[pos] += np.array(sample[4]).copy()
            self.pos += 1

class PrioritizedBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
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
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = prio

    def get_len(self):
        return len(self.buffer)
