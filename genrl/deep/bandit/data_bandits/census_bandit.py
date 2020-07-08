import gzip
import shutil
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class CensusDataBandit(DataBasedBandit):
    def __init__(
        self,
        path: str = "./data/Census/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
    ):
        """
        https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
        """
        super(CensusDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(fpath)
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("USCensus1990.data.txt")
            if Path(path).is_file():
                self.df = pd.read_csv(path)
            else:
                raise FileNotFoundError(
                    "File not found at location {path}, use download flag"
                )

        self.n_actions = len(self.df["dOccup"].unique())
        self.context_columns = [
            i for i in range(self.df.shape[1]) if i != self.df.columns.get_loc("dOccup")
        ]
        self.context_dim = self.df.shape[1] - 1
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        label = self.df["dOccup"].iloc[self.idx]
        r = int(label == (action + 1))
        return r, 1

    def _get_context(self) -> torch.Tensor:
        return torch.tensor(
            self.df.iloc[self.idx, self.context_columns], device=device, dtype=dtype,
        )
