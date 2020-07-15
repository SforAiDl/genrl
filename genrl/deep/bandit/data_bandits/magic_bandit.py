from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class MagicDataBandit(DataBasedBandit):
    """
    https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    """

    def __init__(
        self,
        path: str = "./data/Magic/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
    ):
        super(MagicDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(fpath, header=None)
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("magic04.data")
            if Path(path).is_file():
                self.df = pd.read_csv(path, header=None)
            else:
                raise FileNotFoundError(
                    f"File not found at location {path}, use download flag"
                )

        col = self.df.columns[-1]
        dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False)
        self.df = pd.concat([self.df, dummies.iloc[:, -1]], axis=1)
        self.df = self.df.drop(col, axis=1)

        self.n_actions = 2
        self.context_dim = self.df.shape[1] - 1
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        label = self.df.iloc[self.idx, self.context_dim]
        r = int(label == (action + 1))
        return r, 1

    def _get_context(self) -> torch.Tensor:
        return torch.tensor(
            self.df.iloc[self.idx, : self.context_dim], device=device, dtype=dtype
        )
