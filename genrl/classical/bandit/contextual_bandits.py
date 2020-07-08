from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ContextualBandit(object):
    """
    Base Class for a Multi-armed Bandit

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param context_type: Give context as either tensor or int
    :type bandits: int
    :type arms: int
    :type context_type: str
    """

    def __init__(self, bandits: int = 1, arms: int = 1, context_type: str = "tensor"):
        self._nbandits = bandits
        self._narms = arms
        self.n_actions = arms
        self.context_dim = bandits
        assert (
            context_type == "int" or context_type == "tensor"
        ), f"cotext_type should be either tensor or int, found {context_type}"
        self.context_type = context_type
        self.reset()
        self._regret_hist = []
        self._reward_hist = []

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received at each step
        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def regret_hist(self) -> List[float]:
        """
        Get the history of regrets incurred at each step
        :returns: List of regrest
        :rtype: list
        """
        return self._regret_hist

    @property
    def arms(self) -> int:
        """
        Get the number of arms in each bandit

        :returns: Number of arms in each bandit
        :rtype: int
        """
        return self._narms

    @property
    def bandits(self) -> int:
        """
        Get the number of bandits

        :returns: Number of bandits
        :rtype: int
        """
        return self._nbandits

    def reset(self) -> torch.Tensor:
        """
        Resets the current bandit randomly

        :returns: The current bandit as observation
        :rtype: int
        """
        self.curr_bandit = torch.randint(self.bandits, (1,))
        self.curr_context = F.one_hot(
            self.curr_bandit, num_classes=self.context_dim
        ).to(torch.float)
        if self.context_type == "tensor":
            return self.curr_context
        elif self.context_type == "int":
            return self.curr_bandit.item()

    def step(self, action: int) -> Tuple[Union[int, torch.Tensor], Union[int, float]]:
        """
        Takes an action in the bandit and returns the sampled reward

        This method needs to be implemented in the specific bandit.

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken
        :rtype: int, float ...
        """
        reward, max_reward = self._compute_reward(action)
        self.regret_hist.append(max_reward - reward)
        self.reward_hist.append(reward)
        self.reset()
        if self.context_type == "tensor":
            return self.curr_context.view(-1), reward
        elif self.context_type == "int":
            return self.curr_bandit, reward


class BernoulliCB(ContextualBandit):
    """
    Contextual Bandit with categorial context and bernoulli reward distribution

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param reward_probs: Probabilities of getting rewards
    :type bandits: int
    :type arms: int
    :type reward_probs: numpy.ndarray
    """

    def __init__(
        self,
        bandits: int = 1,
        arms: int = 1,
        reward_probs: np.ndarray = None,
        context_type: str = "tensor",
    ):
        super(BernoulliCB, self).__init__(bandits, arms, context_type)
        if reward_probs is not None:
            self.reward_probs = reward_probs
        else:
            self.reward_probs = np.random.random(size=(bandits, arms))

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a bernoulli distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken and maximum reward
        :rtype: tuple of int
        """
        reward_prob = self.reward_probs[self.curr_bandit, action]
        reward = int(np.random.random() > reward_prob)
        return reward, 1


class GaussianCB(ContextualBandit):
    """
    Contextual Bandit with categorial context and gaussian reward distribution

    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param reward_means: Mean of gaussian distribution for each reward
    :type bandits: int
    :type arms: int
    :type reward_means: numpy.ndarray
    """

    def __init__(
        self,
        bandits: int = 1,
        arms: int = 1,
        reward_means: np.ndarray = None,
        context_type: str = "tensor",
    ):
        super(GaussianCB, self).__init__(bandits, arms, context_type)
        if reward_means is not None:
            self.reward_means = reward_means
        else:
            self.reward_means = np.random.random(size=(bandits, arms))

    def _compute_reward(self, action: int) -> Tuple[float, float]:
        """
        Takes an action in the bandit and returns the sampled reward

        The reward is sampled from a gaussian distribution

        :param action: The action to take
        :type action: int
        :returns: Reward sampled for the action taken and maximum reward
        :rtype: tuples of int
        """
        reward_mean = self.reward_means[self.curr_bandit, action]
        reward = np.random.normal(reward_mean)
        return reward, max(self.reward_means[self.curr_bandit])
