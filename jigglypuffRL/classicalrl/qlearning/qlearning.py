import numpy as np


class QLearning:
    def __init__(self, env, epsilon=0.9, gamma=0.95, lr=0.01):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def get_action(self, s, explore=True):
        if explore == True:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[s, :])

    def update(self, transition):
        s, a, r, s_ = transition

        self.Q[s, a] += self.lr * (
            r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a]
        )
