from typing import List

from ..data_bandits import DataBasedBandit


class DCBAgent:
    def __init__(self, bandit: DataBasedBandit):
        self._bandit = bandit
        self._reward_hist = []

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received for each step
        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    def learn(self, n_timesteps: int):
        context = self._bandit.reset()
        for _ in range(n_timesteps):
            action = self.select_action(context)
            context, reward = self._bandit.step(action)
            self.update_params(context, action, reward)
