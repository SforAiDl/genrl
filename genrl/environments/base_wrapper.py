from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    """
    Base class for all wrappers
    """

    def __init__(self, env, batch_size=None):
        self.env = env
        self._batch_size = batch_size

    # TODO(zeus3101) Add functionality for VecEnvs

    @property
    def batch_size(self):
        """
        The number of batches trained per update
        """
        return self._batch_size

    # TODO(zeus3101) Get get_state, set_state, get_info, get_done methods

    @abstractmethod
    def seed(self, seed=None):
        """
        Set seed for environment
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """
        Render the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Step through the environment

        Must be overriden by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Resets state of environment

        Must be overriden by subclasses

        :returns: Initial state
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Closes environment and performs any other cleanup

        Must be overridden by subclasses
        """
        raise NotImplementedError

    def __enter__(self):
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
