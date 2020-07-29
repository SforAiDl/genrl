import random
from typing import Tuple, Union

import torch


class TransitionDB(object):
    """
    Database for storing (context, action, reward) transitions.

    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        db (dict): Dictionary containing list of transitions.
        db_size (int): Number of transitions stored in database.
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):

        if type(device) is str:
            self.device = (
                torch.device(device)
                if "cuda" in device and torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        self.db = {"contexts": [], "actions": [], "rewards": []}
        self.db_size = 0

    def add(self, context: torch.Tensor, action: int, reward: int):
        """Add (context, action, reward) transition to database

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db["contexts"].append(context)
        self.db["actions"].append(action)
        self.db["rewards"].append(reward)
        self.db_size += 1

    def get_data(
        self, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database

        Args:
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts, actions, rewards tensors.
        """
        if batch_size is None:
            batch_size = self.db_size
        else:
            batch_size = min(batch_size, self.db_size)
        idx = [random.randrange(self.db_size) for _ in range(batch_size)]
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        a = (
            torch.stack([self.db["actions"][i] for i in idx])
            .to(self.device)
            .to(torch.long)
        )
        return x, a, y

    def get_data_for_action(
        self, action: int, batch_size: Union[int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of transition from database for a given action.

        Args:
            action (int): The action to sample transitions for.
            batch_size (Union[int, None], optional): Size of batch required.
                Defaults to None which implies all transitions in the database
                are to be included in batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of stacked
                contexts and rewards tensors.
        """
        action_idx = [i for i in range(self.db_size) if self.db["actions"][i] == action]
        if batch_size is None:
            t_batch_size = len(action_idx)
        else:
            t_batch_size = min(batch_size, len(action_idx))
        idx = random.sample(action_idx, t_batch_size)
        x = (
            torch.stack([self.db["contexts"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
        )
        y = (
            torch.tensor([self.db["rewards"][i] for i in idx])
            .to(self.device)
            .to(torch.float)
            .unsqueeze(1)
        )
        return x, y
