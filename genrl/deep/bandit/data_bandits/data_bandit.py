import urllib.request
from pathlib import Path
from typing import List, Tuple, Union

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def download_data(
    path: str, url: str, force: bool = False, filename: Union[str, None] = None
) -> str:
    """Download data to given location from given URL.

    Args:
        path (str): Location to download to.
        url (str): URL to download file from.
        force (bool, optional): Force download even if file exists. Defaults to False.
        filename (Union[str, None], optional): Name to save file under. Defaults to None
            which implies original filename is to be used.

    Returns:
        str: Path to downloaded file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(url).name
    fpath = path.joinpath(filename)
    if fpath.is_file() and not force:
        return str(fpath)

    try:
        print(f"Downloading {url} to {fpath.resolve()}")
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead.")
            print(f" Downloading {url} to {path}")
            urllib.request.urlretrieve(url, fpath)
        else:
            raise e

    return str(fpath)


class DataBasedBandit(object):
    """Base class for contextual bandits based on  datasets.
    """

    def __init__(self):
        self._reset()

    @property
    def reward_hist(self) -> List[float]:
        """History of rewards generated.

        Returns:
            List[float]: List of rewards.
        """
        return self._reward_hist

    @property
    def regret_hist(self) -> List[float]:
        """History of regrets generated.

        Returns:
            List[float]: List of regrets.
        """
        return self._regret_hist

    @property
    def cum_reward_hist(self) -> Union[List[int], List[float]]:
        """History of rewards generated.

        Returns:
            List[float]: List of rewards.
        """
        return self._cum_regret_hist

    @property
    def cum_regret_hist(self) -> Union[List[int], List[float]]:
        """History of cumulative regrets.

        Returns:
            List[float]: List of cumulative regrets.
        """
        return self._cum_reward_hist

    @property
    def cum_regret(self) -> Union[int, float]:
        """Cumulative regret.

        Returns:
            Union[int, float]: Cumulative regret.
        """
        return self._cum_regret

    @property
    def cum_reward(self) -> Union[int, float]:
        """Cumulative reward.

        Returns:
            Union[int, float]: Cumulative reward.
        """
        return self._cum_reward

    def _reset(self):
        """Resets tracking metrics.
        """
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
