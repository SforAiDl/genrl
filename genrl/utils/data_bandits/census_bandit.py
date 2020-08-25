from typing import Tuple

import pandas as pd
import torch

from genrl.utils.data_bandits.base import DataBasedBandit
from genrl.utils.data_bandits.utils import download_data, fetch_data_with_header

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"


class CensusDataBandit(DataBasedBandit):
    """A contextual bandit based on the Census dataset.

    Source:
        https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Census/".
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
        super(CensusDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Census/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            fpath = download_data(path, url, force_download)
            self._df = pd.read_csv(fpath)
        else:
            self._df = fetch_data_with_header(path, "USCensus1990.data.txt")

        self.n_actions = len(self._df["dOccup"].unique())
        self._context_columns = [
            i
            for i in range(self._df.shape[1])
            if i != self._df.columns.get_loc("dOccup")
        ]
        self.context_dim = self._df.shape[1] - 1
        self.len = len(self._df)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self.df = self._df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        label = self._df["dOccup"].iloc[self.idx]
        r = int(label == (action + 1))
        return r, 1

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self._df.iloc[self.idx, self._context_columns],
            device=self.device,
            dtype=torch.float,
        )
