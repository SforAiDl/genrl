from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from genrl.utils.data_bandits.base import DataBasedBandit
from genrl.utils.data_bandits.utils import download_data

URL = "https://storage.googleapis.com/bandits_datasets/raw_stock_contexts"


class FinancialDataBandit(DataBasedBandit):
    """A contextual bandit based on Financial Stock data.

    Source:
        https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Financial/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.
        device (torch.device): Device to use for tensor operations.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(self, **kwargs):
        super(FinancialDataBandit, self).__init__(kwargs.get("device", "cpu"))

        self.n_actions = 8

        path = kwargs.get("path", "./data/Financial/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(
                fpath, header=None, skiprows=[0], sep=" ", dtype=np.float32
            ).dropna()
        else:
            fpath = Path(path).joinpath("raw_stock_contexts")
            self.df = pd.read_csv(
                fpath, header=None, skiprows=[0], sep=" ", dtype=np.float32
            ).dropna()

        self.context_dim = self.df.shape[1]
        self.len = len(self.df)

        self._generate_rewards()

    def _generate_rewards(self):
        # Vector with additive noise levels for each action
        noise_stds = [0.01 * (i + 1) for i in range(self.n_actions)]
        betas = np.random.uniform(-1, 1, (self.context_dim, self.n_actions))
        betas /= np.linalg.norm(betas, axis=0)

        mean_rewards = np.dot(self.df, betas)
        noise = np.random.normal(scale=noise_stds, size=mean_rewards.shape)

        self.rewards = mean_rewards + noise
        self.max_rewards = np.max(self.rewards, axis=1)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self._generate_rewards()
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        r = self.rewards[self.idx, action]
        max_r = self.max_rewards[self.idx]
        return r, max_r

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self.df.iloc[self.idx],
            device=self.device,
            dtype=torch.float,
        )
