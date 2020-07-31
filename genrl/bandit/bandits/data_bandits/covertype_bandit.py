import gzip
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from genrl.bandit.bandits.data_bandits.base import DataBasedBandit
from genrl.bandit.bandits.data_bandits.utils import download_data

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)


class CovertypeDataBandit(DataBasedBandit):
    """A contextual bandit based on the Covertype dataset.

    Source:
        https://archive.ics.uci.edu/ml/datasets/covertype

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Covertype/".
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
        super(CovertypeDataBandit, self).__init__(
            "Covertype", kwargs.get("device", "cpu")
        )

        self.dir = kwargs.get("path", self.dir)
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            gz_fpath = download_data(self.dir, url, force_download)
        else:
            gz_fpath = self.dir.joinpath("covtype.data.gz")

        with gzip.open(gz_fpath, "rb") as f_in:
            fpath = Path(gz_fpath).parent.joinpath("covtype.data")
            with open(fpath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        self._df = pd.read_csv(fpath, header=None, na_values=["?"]).dropna()
        fpath.unlink()

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
        r = int(label == (action + 1))
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
