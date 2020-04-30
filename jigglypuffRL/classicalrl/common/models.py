import numpy as np


class TabularModel:
    """
    Sample-based tabular model class for deterministic, discrete environments
    :param s_dim: (int) environment state dimension
    :param a_dim: (int) environment action dimension
    """

    def __init__(self, s_dim, a_dim):
        self.s_model = np.zeros((s_dim, a_dim), dtype=np.uint8)
        self.r_model = np.zeros((s_dim, a_dim))

    def add(self, s, a, r, s_):
        self.s_model[s, a] = s_
        self.r_model[s, a] = r

    def sample(self):
        # select random visited state
        s = np.random.choice(np.where(np.sum(self.s_model, axis=1) > 0)[0])
        # random action in that state
        a = np.random.choice(np.where(self.s_model[s] > 0)[0])
        return s, a

    def step(self, s, a):
        r = self.r_model[s, a]
        s_ = self.s_model[s, a]
        return r, s_


model_registry = {"tabular": TabularModel}


def get_model_from_name(name_):
    """
    get model object from name
    :param name_: (str) name of the model ['tabular']
    """
    return model_registry[name_]
