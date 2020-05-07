import numpy as np


class TabularModel:
    """
    Sample-based tabular model class for deterministic, discrete environments
    :param s_dim: (int) environment state dimension
    :param a_dim: (int) environment action dimension
    """

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.s_model = np.zeros((s_dim, a_dim), dtype=np.uint8)
        self.r_model = np.zeros((s_dim, a_dim))

    def add(self, s, a, r, s_):
        """
        add transition to model
        :param s: (float array) state
        :param a: (int) action
        :param r: (int) reward
        :param s_: (float array) next state
        """
        self.s_model[s, a] = s_
        self.r_model[s, a] = r

    def sample(self):
        """
        sample state action pair from model
        :returns s, a: state and action
        """
        # select random visited state
        s = np.random.choice(np.where(np.sum(self.s_model, axis=1) > 0)[0])
        # random action in that state
        a = np.random.choice(np.where(self.s_model[s] > 0)[0])
        return s, a

    def step(self, s, a):
        """
        return consequence of action at state
        :returns r, s_: reward and next state
        """
        r = self.r_model[s, a]
        s_ = self.s_model[s, a]
        return r, s_

    def is_empty(self):
        """
        True if model not updated yet
        """
        return not (np.any(self.s_model) or np.any(self.r_model))


model_registry = {"tabular": TabularModel}


def get_model_from_name(name_):
    """
    get model object from name
    :param name_: (str) name of the model ['tabular']
    """
    if name_ in model_registry:
        return model_registry[name_]
    return NotImplementedError
