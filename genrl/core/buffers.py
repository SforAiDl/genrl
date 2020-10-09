import os
import random
from collections import deque
from typing import NamedTuple, Tuple

import torch


class BaseBuffer(object):
    """Base class that represents a buffer (rollout or replay)

    Attributes:
        buffer_size (int): Max number of elements in the buffer
    """

    def __init__(self, buffer_size: int):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

    @staticmethod
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor:
        """Swap and Flatten method

        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        Args:
            arr (:obj:`torch.Tensor`): Array to modify

        Returns:
            new_arr (:obj:`torch.Tensor`): Modified Array
        """
        shape = arr.shape
        if len(shape) < 3:
            arr = arr.unsqueeze(-1)
            shape = shape + (1,)

        return arr.permute(1, 0, *(torch.arange(2, len(shape)))).reshape(
            shape[0] * shape[1], *shape[2:]
        )

    def size(self) -> int:
        """Returns size of the buffer

        Returns:
            size (int): The current size of the buffer
        """
        raise NotImplementedError

    def add(self, *args, **kwargs) -> None:
        """Adds elements to the buffer"""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the buffer"""
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int) -> Tuple:
        """Sample from the buffer

        Args:
            batch_size (int): Number of element to sample

        Returns:
            samples (:obj:`namedtuple`): Named tuple of the sampled experiences
        """
        raise NotImplementedError

    def save(self, directory: str = None, run_num: int = None) -> None:
        """Saves the buffer locally

        The buffers are saved locally so they can be used as a dataset for
            Offline RL or for other purposes

        Args:
            directory (string): Directory to save buffers in
            run_num (int): The run number associated with the training run
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Loads the buffer from the file

        Args:
            path (str): Path of the pickled file of the buffer replays/rollouts
        """
        if not os.path.exists(path):
            raise FileNotFoundError

        raise NotImplementedError


class ReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer(BaseBuffer):
    """Vanilla Experience Replay Buffer

    Attributes:
        buffer_size (int): Max number of element in the buffer
        device (:obj:`torch.device` or str):  PyTorch device to which the values will be converted
    """

    def __init__(self, *args, **kwargs):
        super(ReplayBuffer, self).__init__(*args, **kwargs)
        self.buffer = deque([], maxlen=self.buffer_size)

    def size(self) -> int:
        """Returns size of the buffer

        Returns:
            size (int): The current size of the buffer
        """
        return len(self.buffer)

    def add(self, experience: Tuple) -> None:
        """Adds elements to the buffer

        Args:
            experience (:obj:`tuple`): Tuple containing state, action, reward, next_state and done
        """
        self.buffer.append(experience)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the buffer

        Args:
            batch_size (int): Number of element to sample

        Returns:
            samples (:obj:`namedtuple`): Named tuple of the sampled experiences
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        return ReplayBufferSamples(states, actions, rewards, next_states, dones)

    def save(self, directory: str = None, run_num: int = None) -> None:
        """Saves the buffer locally

        The buffers are saved locally so they can be used as a dataset for
            Offline RL or for other purposes

        Args:
            directory (string): Directory to save buffers in
            run_num (int): The run number associated with the training run
        """
        if run_num is None:
            if list(os.scandir(directory)) == []:
                run_num = 0
            else:
                last_path = sorted(
                    os.scandir(directory), key=lambda d: d.stat().st_mtime
                )[-1].path
                run_num = int(last_path[len(directory) + 1 :].split("-")[0]) + 1

        torch.save(self.buffer, "{}/run-{}.pt".format(directory, run_num))

    def load(self, path: str) -> None:
        """Loads the buffer from the file

        Args:
            path (str): Path of the pickled file of the buffer replays/rollouts
        """
        self.buffer = torch.load(path)


class PrioritizedReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class PrioritizedBuffer(BaseBuffer):
    """Prioritized Experience Replay Mechanism

    Attributes:
        buffer_size (int): Max number of element in the buffer
        alpha (float): Level of prioritization
        beta (float): Bias factor used to correct IS Weights
        device (:obj:`torch.device` or str):  PyTorch device to which the values will be converted
    """

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super(PrioritizedBuffer, self).__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.buffer = deque([], maxlen=self.buffer_size)
        self.priorities = deque([], maxlen=self.buffer_size)

    def add(self, experience: Tuple) -> None:
        """Adds elements to the buffer

        Args:
            experience (:obj:`tuple`): Tuple containing state, action, reward, next_state and done
        """
        self.buffer.append(experience)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int
    ) -> (
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Sample from the buffer

        Args:
            batch_size (int): Number of element to sample

        Returns:
            samples (:obj:`namedtuple`): Named tuple of the sampled experiences
        """
        total = len(self.buffer)
        priorities = torch.FloatTensor(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = torch.multinomial(probabilities, batch_size)

        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = [self.buffer[i] for i in indices]
        (states, actions, rewards, next_states, dones) = map(torch.stack, zip(*samples))

        return PrioritizedReplayBufferSamples(
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        )

    def update_priorities(self, batch_indices: Tuple, batch_priorities: Tuple) -> None:
        """
                Updates list of priorities with new order of priorities

                :param batch_indices: List of indices of batch
                :param batch_priorities: (List of priorities of the batch at the
        specific indices)
                :type batch_indices: list or tuple
                :type batch_priorities: list or tuple
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = priority.mean()

    def size(self) -> int:
        """Returns size of the buffer

        Returns:
            size (int): The current size of the buffer
        """
        return len(self.buffer)
