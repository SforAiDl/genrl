from typing import List, Tuple, Union

import numpy as np

from genrl.core.bandit import BanditAgent, MultiArmedBandit


class MABAgent(BanditAgent):
    """
    Base Class for Contextual Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of Q values is required
    :type bandit: MultiArmedlBandit type object
    """

    def __init__(self, bandit: MultiArmedBandit):
        super(MABAgent, self).__init__()

        self._bandit = bandit
        assert bandit.context_type == "int", (
            f"Context type of bandit should be int, " f"found {bandit.context_type}"
        )
        self._regret = 0.0
        self._regret_hist = []
        self._action_hist = []
        self._reward_hist = []
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def action_hist(self) -> Tuple[int, int]:
        """
        Get the history of actions taken for contexts

        :returns: List of context, actions pairs
        :rtype: list
        """
        return self._action_hist

    @property
    def regret(self) -> float:
        """
        Get the current regret

        :returns: The current regret
        :rtype: float
        """
        return self._regret

    @property
    def regret_hist(self) -> List[float]:
        """
        Get the history of regrets incurred for each step

        :returns: List of rewards
        :rtype: list
        """
        return self._regret_hist

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received for each step

        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def counts(self) -> np.ndarray:
        """
        Get the number of times each action has been taken

        :returns: Numpy array with count for each action
        :rtype: numpy.ndarray
        """
        return self._counts

    def select_action(self, context: int) -> int:
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param context: the context to select action for
        :type context: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(
        self, context: int, action: int, reward: Union[int, float]
    ) -> None:
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: int or float
        """
        raise NotImplementedError
