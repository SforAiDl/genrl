from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


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

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(
        self,
        path: str = "./data/Census/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
    ):
        super(CensusDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            fpath = download_data(path, url, force_download)
            self._df = pd.read_csv(fpath)
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("USCensus1990.data.txt")
            if Path(path).is_file():
                self._df = pd.read_csv(path)
            else:
                raise FileNotFoundError(
                    f"File not found at location {path}, use download flag"
                )

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
            self._df.iloc[self.idx, self._context_columns], device=device, dtype=dtype,
        )
