import numpy as np
from typing import Any, Union


class BaseValue:
    """
    MLP Value Function class

    :param state_dim: State dimensions of environment
    :param action_dim: Action dimensions of environment
    :type state_dim: int
    :type action_dim: int
    """

    def __init__(
        self, state_dim: int, action_dim: int,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __call__(self, state: Any, action: Any) -> float:
        """
        Gets the value estimate for given state and action

        :param state: state
        :param action: action
        :type state: int
        :type action: int
        :returns: value
        :rtype: float
        """

        raise NotImplementedError

    def update(self, state: Any, action: Any, reward: Union[int, float]) -> None:
        """
        Updates the value estimate

        :param state: state
        :param action: action
        :param reward: reward
        :type state: int
        :type action: int
        :type reward: int or float
        """


value_registry = {}


def get_value_from_name(name_: str) -> None:
    """
    Gets the value function given the name of the value function

    :param name_: Name of the value function needed
    :type name_: string
    :returns: Value function
    """
    if name_ in value_registry:
        return value_registry[name_]
    raise NotImplementedError
