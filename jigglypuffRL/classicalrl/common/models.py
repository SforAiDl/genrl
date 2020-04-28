import numpy as np

class TabularModel:
    def __init__(self, s_dim, a_dim):
        self.s_model = np.zeros((s_dim, a_dim), dtype=np.uint8)
        self.r_model = np.zeros((s_dim, a_dim))

    def add(self, s, a, s_, r):
        self.s_model[s, a] = s_
        self.rewards[s,a] = r

    def sample(self):
        # select random visited state
        s = np.random.choice(np.where(np.sum(self.s_model, axis=1) > 0)[0])
        # random action in that state
        a = np.random.choice(np.where(self.s_model[s] > 0)[0])

    def step(self, s, a):
        s_ = self.s_model[s,a]
        r = self.s_model[s,a]
        return s_, r
