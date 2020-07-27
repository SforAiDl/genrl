from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch

from genrl.bandit.bandits.data_bandits.base import DataBasedBandit
from genrl.bandit.bandits.data_bandits.utils import download_data

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"


class MushroomDataBandit(DataBasedBandit):
    """A contextual bandit based on the Mushroom dataset.

    Source:
        https://archive.ics.uci.edu/ml/datasets/Mushroom

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Magic/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.
        r_pass (int): Reward generated for passing. Defaults to 0
        r_edible (int): Reward generated for eating an edible mushroom. Defaults to 5
        r_poisonous_lucky (int): Reward generated for eating a poisonous mushroom
            and getting lucky. Defaults to 5
        r_poisonous_unlucky (int): Reward generated for eating a poisonous mushroom
            and getting unlucky. Defaults to -35
        lucky_prob (float): Probability with which you can get lucky when eating a
            poisonous mushroom. Defaults to 0.5
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
        device: str = "cpu",
    ):
        super(MushroomDataBandit, self).__init__(device)

        if download:
            if url is None:
                url = URL
            fpath = download_data(path, url, force_download)
            self.df = pd.read_csv(fpath, header=None)
        else:
            if Path(path).is_dir():
                path = Path(path).joinpath("agaricus-lepiota.data")
            if Path(path).is_file():
                self.df = pd.read_csv(path, header=None)
            else:
                raise FileNotFoundError(
                    f"File not found at location {path}, use download flag"
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
        """Reset bandit by shuffling indices and get new context.

        Returns:
            torch.Tensor: Current context selected by bandit.
        """
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
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns:
            Tuple[int, int]: Computed reward.
        """
        if action == 0:
            r = self.r_pass
        elif action == 1:
            r = self.eat_rewards[self.idx]
        else:
            raise ValueError(f"Action {action} undefined for mushroom data bandit")
        return r, self.optimal_exp_rewards[self.idx]

    def _get_context(self) -> torch.Tensor:
        """Get the vector for current selected context.

        Returns:
            torch.Tensor: Current context vector.
        """
        return torch.tensor(
            self.df.iloc[self.idx, self.n_actions :],
            device=self.device,
            dtype=torch.float,
        )
