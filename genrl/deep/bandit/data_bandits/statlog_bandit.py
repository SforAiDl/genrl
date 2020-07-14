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
                    "File not found at location {path}, use download flag"
                )

        self.n_actions = len(self.df.iloc[:, -1].unique())
        self.context_dim = self.df.shape[1] - 1
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        label = self.df.iloc[self.idx, self.context_dim]
        r = int(label == action)
        return r, 1

    def _get_context(self) -> torch.Tensor:
        return torch.tensor(
            self.df.iloc[self.idx, : self.context_dim], device=device, dtype=dtype
        )
