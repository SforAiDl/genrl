import subprocess
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class StatlogDataBandit(DataBasedBandit):
    """A contextual bandit based on the Statlog (Shuttle) dataset.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Magic/".
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
        path: str = "./data/Statlog/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
    ):
        super(StatlogDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            z_fpath = download_data(path, url, force_download)
            subprocess.run(["uncompress", "-f", "-k", z_fpath])
            fpath = Path(z_fpath).parent.joinpath("shuttle.trn")
            self.df = pd.read_csv(fpath, header=None, delimiter=" ")
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("shuttle.trn")
            if Path(path).is_file():
                self.df = pd.read_csv(path, header=None, delimiter=" ")
            else:
                raise FileNotFoundError(
                    f"File not found at location {path}, use download flag"
                )

        self.n_actions = len(self.df.iloc[:, -1].unique())
        self.context_dim = self.df.shape[1] - 1
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        label = self.df.iloc[self.idx, self.context_dim]
        r = int(label == action)
        return r, 1

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self.df.iloc[self.idx, : self.context_dim], device=device, dtype=dtype
        )
