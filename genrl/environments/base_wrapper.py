from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseWrapper(ABC):
    """
    Base class for all wrappers
    """

    def __init__(self, env: Any, batch_size: int = None):
        self.env = env
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """
        The number of batches trained per update
        """
        return self._batch_size

    @abstractmethod
    def seed(self, seed: int = None) -> None:
        """
        Set seed for environment
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        """
        Render the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """
        Step through the environment

        Must be overriden by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Resets state of environment

        Must be overriden by subclasses

        :returns: Initial state
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Closes environment and performs any other cleanup

        Must be overridden by subclasses
        """
        raise NotImplementedError

    def __enter__(self) -> None:
        """
        Allows environment to be called using with statement
        """
        return self

    def __exit__(self, exec_type, exec_value, exec_traceback):
        """
        Allows environment to be called using with statement
        Arguments are necessary to make the method callable
        """
        self.close()
