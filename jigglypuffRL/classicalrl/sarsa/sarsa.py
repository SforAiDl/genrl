import numpy as np


class SARSA:
    def __init__(self, env, epsilon=0.9, lmbda=0.9, gamma=0.95, lr=0.01):
        self.env = env
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        self.e = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def get_action(self, s, explore=True):
        if explore == True:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[s, :])

    def update(self, transition):
        s, a, r, s_ = transition

        a_ = self.get_action(s_)
        delta = r + self.gamma * self.Q[s_, a_] - self.Q[s, a]
        self.e[s, a] += 1

        for si in range(self.env.observation_space.n):
            for ai in range(self.env.action_space.n):
                self.Q[s, a] += self.lr * delta * self.e[s, a]
                self.e[s, a] = self.gamma * self.lmbda * self.e[s, a]
