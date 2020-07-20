import gzip
import shutil
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


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

    Attributes:
        n_actions (int): Number of actions available.
        context_dim (int): The length of context vector.
        len (int): The number of examples (context, reward pairs) in the dataset.

    Raises:
        FileNotFoundError: If file is not found at specified path.
    """

    def __init__(
        self,
        path: str = "./data/Covertype/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
    ):
        super(CovertypeDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            gz_fpath = download_data(path, url, force_download)
            with gzip.open(gz_fpath, "rb") as f_in:
                fpath = Path(gz_fpath).parent.joinpath("covtype.data")
                with open(fpath, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self._df = pd.read_csv(fpath, header=None, na_values=["?"]).dropna()
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("covtype.data")
            if Path(path).is_file():
                self._df = pd.read_csv(path, header=None, na_values=["?"]).dropna()
            else:
                raise FileNotFoundError(
                    f"File not found at location {path}, use download flag"
                )
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
            self._df.iloc[self.idx, : self.context_dim], device=device, dtype=dtype
        )
