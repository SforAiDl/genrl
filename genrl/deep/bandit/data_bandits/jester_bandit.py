import zipfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = [
    "http://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip",
    "http://eigentaste.berkeley.edu/dataset/jester_dataset_1_2.zip",
    "http://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class AdultDataBandit(DataBasedBandit):
    def __init__(
        self,
        path: str = "./data/Adult/",
        download: bool = False,
        force_download: bool = False,
        url: Union[List[str], None] = None,
    ):
        super(AdultDataBandit, self).__init__()

        dfs = []
        if download:
            if url is None:
                url = URL
            for i, u in enumerate(URL):
                zpath = download_data(path, url, force_download)
                fpath = zipfil.ZipFile(zpath).extract(f"jester-data-{i}.xls")
                dfs.append(pd.read_excel(fpath, encoding="sys.getfilesystemencoding()"))
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("adult.data")
            if Path(path).is_file():
                self.df = pd.read_csv(path, header=None, na_values=["?", " ?"]).dropna()
            else:
                raise FileNotFoundError(
                    "File not found at location {path}, use download flag"
                )
        self.df = pd.concat(dfs, axis=0)

        for col in self.df.columns[[1, 3, 5, 6, 7, 8, 9, 13, 14]]:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df = self.df.drop(col, axis=1)

        print(list(self.df.columns))
        self.df[self.df.columns[-2]] += self.df[self.df.columns[-1]]
        self.df.drop(self.df.columns[-1], axis=1)

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
