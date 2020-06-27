from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from .data_bandit import DataBasedBandit, download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class MushroomDataBandit(DataBasedBandit):
    def __init__(
        self,
        path: str = "./data/Mushroom/",
        download: bool = False,
        force_download: bool = False,
        url: Union[str, None] = None,
        r_pass: int = 0,
        r_edible: int = 5,
        r_poisonous_lucky: int = 5,
        r_poisonous_unlucky: int = -35,
        lucky_prob: float = 0.5,
    ):
        super(MushroomDataBandit, self).__init__()

        if download:
            if url is None:
                url = URL
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(fpath)
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("agaricus-lepiota.data")
            if Path(path).is_file():
                self.df = pd.read_csv(path, header=None)
            else:
                raise FileNotFoundError(
                    "File not found at location {path}, use download flag"
                )

        for col in self.df.columns:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df = self.df.drop(col, axis=1)

        self.r_pass = r_pass
        self.r_edible = r_edible
        self.r_poisonous_lucky = r_poisonous_lucky
        self.r_poisonous_unlucky = r_poisonous_unlucky
        self.lucky_prob = lucky_prob
        self.r_poisonous_exp = r_poisonous_lucky * lucky_prob + r_poisonous_unlucky * (
            1 - lucky_prob
        )
        self.n_actions = 2
        self.context_dim = self.df.shape[1] - 2
        self.len = len(self.df)

    def reset(self) -> torch.Tensor:
        self._reset()
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        poison_rewards = np.random.choice(
            [self.r_poisonous_lucky, self.r_poisonous_unlucky],
            p=[self.lucky_prob, 1 - self.lucky_prob],
            size=len(self.df),
        )
        self.eat_rewards = self.r_edible * self.df.iloc[:, 0] + np.multiply(
            poison_rewards, self.df.iloc[:, 1]
        )
        self.optimal_exp_rewards = (
            self.r_edible * self.df.iloc[:, 0]
            + max(self.r_pass, self.r_poisonous_exp) * self.df.iloc[:, 1]
        )
        if self.r_pass > self.r_poisonous_exp:
            self.optimal_actions = self.df.iloc[:, 0].to_numpy()
        else:
            self.optimal_actions = np.ones(len(self.df))

        return self._get_context()

    def _compute_reward(self, action: int) -> Tuple[int, int]:
        label = list(self.df.iloc[self.idx, : self.n_actions])
        if action == 0:
            r = self.r_pass
        elif action == 1:
            r = self.eat_rewards[self.idx]
        return r, self.optimal_exp_rewards[self.idx]

    def _get_context(self) -> torch.Tensor:
        return torch.tensor(
            self.df.iloc[self.idx, self.n_actions :], device=device, dtype=dtype
        )
