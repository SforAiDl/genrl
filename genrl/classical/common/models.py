import numpy as np


class TabularModel:
    """
    Sample-based tabular model class for deterministic, discrete environments

    :param s_dim: environment state dimension
    :param a_dim: environment action dimension
    :type s_dim: int
    :type a_dim: int
    """

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.s_model = np.zeros((s_dim, a_dim), dtype=np.uint8)
        self.r_model = np.zeros((s_dim, a_dim))

    def add(self, s, a, r, s_):
        """
        add transition to model
        :param s: state
        :param a: action
        :param r: reward
        :param s_: next state
        :type s: float array
        :type a: int
        :type r: int
        :type s_: float array
        """
        self.s_model[s, a] = s_
        self.r_model[s, a] = r

    def sample(self):
        """
        sample state action pair from model

        :returns: state and action
        :rtype: int, float, ... ; int, float, ...
        """
        # select random visited state
        s = np.random.choice(np.where(np.sum(self.s_model, axis=1) > 0)[0])
        # random action in that state
        a = np.random.choice(np.where(self.s_model[s] > 0)[0])
        return s, a

    def step(self, s, a):
        """
        return consequence of action at state

        :returns: reward and next state
        :rtype: int; int, float, ...
        """
        r = self.r_model[s, a]
        s_ = self.s_model[s, a]
        return r, s_

    def is_empty(self):
        """
        Check if the model has been updated or not 

        :returns: True if model not updated yet
        :rtype: bool
        """
        return not (np.any(self.s_model) or np.any(self.r_model))


model_registry = {"tabular": TabularModel}


def get_model_from_name(name_):
    """
    get model object from name

    :param name_: name of the model ['tabular']
    :type name_: str
    :returns: the model
    """
    if name_ in model_registry:
        return model_registry[name_]
    return NotImplementedError
