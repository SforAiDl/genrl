import urllib.request
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def download_data(
    path: str, url: str, force: bool = False, filename: Union[str, None] = None
) -> str:
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
    def __init__(self):
        self._reset()

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

    def _reset(self) -> torch.Tensor:
        self.idx = 0
        self._reward_hist = []
        self._regret_hist = []

    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        reward, max_reward = self._compute_reward(action)
        self.regret_hist.append(max_reward - reward)
        self.reward_hist.append(reward)
        self.idx += 1
        if not self.idx < self.len:
            self.idx = 0
        context = self._get_context()
        return context, reward

    def _step(self, action: int) -> Tuple[torch.Tensor, int, int]:
        raise NotImplementedError

    def _get_context(self) -> torch.Tensor:
        raise NotImplementedError
