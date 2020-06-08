from typing import Tuple

import numpy as np


class TabularModel:
    """
    Sample-based tabular model class for deterministic, discrete environments

    :param s_dim: environment state dimension
    :param a_dim: environment action dimension
    :type s_dim: int
    :type a_dim: int
    """

    def __init__(self, s_dim: int, a_dim: int):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.s_model = np.zeros((s_dim, a_dim), dtype=np.uint8)
        self.r_model = np.zeros((s_dim, a_dim))

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        """
        add transition to model
        :param state: state
        :param action: action
        :param reward: reward
        :param next_state: next state
        :type state: float array
        :type action: int
        :type reward: int
        :type next_state: float array
        """
        self.s_model[state, action] = next_state
        self.r_model[state, action] = reward

    def sample(self) -> Tuple:
        """
        sample state action pair from model

        :returns: state and action
        :rtype: int, float, ... ; int, float, ...
        """
        # select random visited state
        state = np.random.choice(np.where(np.sum(self.s_model, axis=1) > 0)[0])
        # random action in that state
        action = np.random.choice(np.where(self.s_model[state] > 0)[0])
        return state, action

    def step(self, state: np.ndarray, action: np.ndarray) -> Tuple:
        """
        return consequence of action at state

        :returns: reward and next state
        :rtype: int; int, float, ...
        """
        reward = self.r_model[state, action]
        next_state = self.s_model[state, action]
        return reward, next_state

    def is_empty(self) -> bool:
        """
        Check if the model has been updated or not

        :returns: True if model not updated yet
        :rtype: bool
        """
        return not (np.any(self.s_model) or np.any(self.r_model))


model_registry = {"tabular": TabularModel}


def get_model_from_name(name_: str):
    """
    get model object from name

    :param name_: name of the model ['tabular']
    :type name_: str
    :returns: the model
    """
    if name_ in model_registry:
        return model_registry[name_]
    return NotImplementedError
