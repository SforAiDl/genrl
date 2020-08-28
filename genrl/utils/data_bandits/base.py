from typing import List, Tuple, Union

import torch

from genrl.core.bandit import Bandit


class DataBasedBandit(Bandit):
    """Base class for contextual bandits based on  datasets.

    Args:
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, device: str = "cpu", **kwargs):

        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self._reset()

    @property
    def reward_hist(self) -> List[float]:
        """List[float]: History of rewards generated."""
        return self._reward_hist

    @property
    def regret_hist(self) -> List[float]:
        """List[float]: History of regrets generated."""
        return self._regret_hist

    @property
    def cum_reward_hist(self) -> Union[List[int], List[float]]:
        """List[float]: History of cumulative rewards generated."""
        return self._cum_regret_hist

    @property
    def cum_regret_hist(self) -> Union[List[int], List[float]]:
        """List[float]: History of cumulative regrets generated."""
        return self._cum_reward_hist

    @property
    def cum_regret(self) -> Union[int, float]:
        """Union[int, float]: Cumulative regret."""
        return self._cum_regret

    @property
    def cum_reward(self) -> Union[int, float]:
        """Union[int, float]: Cumulative reward."""
        return self._cum_reward

    def _reset(self):
        """Resets tracking metrics."""
        self.idx = 0
        self._cum_regret = 0
        self._cum_reward = 0
        self._reward_hist = []
        self._regret_hist = []
        self._cum_regret_hist = []
        self._cum_reward_hist = []

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Note:
            This method needs to be implemented in the specific bandit.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        raise NotImplementedError

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Note:
            This method needs to be implemented in the specific bandit.

        Returns:
            torch.Tensor: Current context vector.
        """
        raise NotImplementedError

    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        """Generate reward for given action and select next context.

        This method also updates the various regret and reward trackers
        as well the current context index.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[torch.Tensor, int]: Tuple of the next context and the
                reward generated for given action
        """
        reward, max_reward = self._compute_reward(action)
        regret = max_reward - reward
        self._cum_regret += regret
        self.cum_regret_hist.append(self._cum_regret)
        self.regret_hist.append(regret)
        self._cum_reward += reward
        self.cum_reward_hist.append(self._cum_reward)
        self.reward_hist.append(reward)
        self.idx += 1
        if not self.idx < self.len:
            self.idx = 0
        context = self._get_context()
        return context, reward

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Note:
            This method needs to be implemented in the specific bandit.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        raise NotImplementedError
