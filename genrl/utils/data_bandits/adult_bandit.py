from typing import Tuple

import pandas as pd
import torch

from genrl.utils.data_bandits.base import DataBasedBandit
from genrl.utils.data_bandits.utils import download_data, fetch_data_without_header

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


class AdultDataBandit(DataBasedBandit):
    """A contextual bandit based on the Adult dataset.

    Source:
        http://archive.ics.uci.edu/ml/datasets/Adult

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Adult/".
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
        super(AdultDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Adult/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            self._df = pd.read_csv(fpath, header=None, na_values=["?", " ?"]).dropna()
        else:
            self._df = fetch_data_without_header(
                path, "adult.data", na_values=["?", " ?"]
            )

        for col in self._df.columns[[1, 3, 5, 6, 7, 8, 9, 13, 14]]:
            dummies = pd.get_dummies(self._df[col], prefix=col, drop_first=False)
            self._df = pd.concat([self._df, dummies], axis=1)
            self._df = self._df.drop(col, axis=1)

        self._df[self._df.columns[-2]] += self._df[self._df.columns[-1]]
        self._df.drop(self._df.columns[-1], axis=1)

        self.n_actions = len(self._df.iloc[:, -1].unique())
        self.context_dim = self._df.shape[1] - 1
        self.len = len(self._df)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self._df = self._df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        label = self._df.iloc[self.idx, self.context_dim]
        r = int(label == action)
        return r, 1

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self._df.iloc[self.idx, : self.context_dim],
            device=self.device,
            dtype=torch.float,
        )
