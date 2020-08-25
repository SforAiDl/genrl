from typing import Optional

import torch

from genrl.core import Bandit, BanditAgent


class DCBAgent(BanditAgent):
    """Base class for deep contextual bandit solving agents

    Args:
        bandit (gennav.deep.bandit.data_bandits.DataBasedBandit): The bandit to solve
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        bandit (gennav.deep.bandit.data_bandits.DataBasedBandit): The bandit to solve
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, bandit: Bandit, device: str = "cpu", **kwargs):

        super(DCBAgent, self).__init__()

        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self._bandit = bandit
        self.context_dim = self._bandit.context_dim
        self.n_actions = self._bandit.n_actions
        self._action_hist = []
        self.init_pulls = kwargs.get("init_pulls", 3)

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context

        Args:
            context (torch.Tensor): The context vector to select action for

        Note:
            This method needs to be implemented in the specific agent.

        Returns:
            int: The action to take
        """
        raise NotImplementedError

    def update_parameters(
        self,
        action: Optional[int] = None,
        batch_size: Optional[int] = None,
        train_epochs: Optional[int] = None,
    ) -> None:
        """Update parameters of the agent.

        Args:
            action (Optional[int], optional): Action to update the parameters for. Defaults to None.
            batch_size (Optional[int], optional): Size of batch to update parameters with. Defaults to None.
            train_epochs (Optional[int], optional): Epochs to train neural network for. Defaults to None.

        Note:
            This method needs to be implemented in the specific agent.

        """
        raise NotImplementedError
