from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Bandit(ABC):
    """Abstract Base class for bandits"""

    @abstractmethod
    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        """Generate reward for given action and select next context.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[torch.Tensor, int]: Tuple of the next context and the
                reward generated for given action
        """

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset bandit.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """


class BanditAgent(ABC):
    """Abstract Base class for bandit solving agents"""

    @abstractmethod
    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context

        Args:
            context (torch.Tensor): The context vector to select action for

        Returns:
            int: The action to take
        """
