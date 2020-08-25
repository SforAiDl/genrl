import subprocess
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from genrl.utils.data_bandits.base import DataBasedBandit
from genrl.utils.data_bandits.utils import download_data, fetch_data_without_header

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z"


class StatlogDataBandit(DataBasedBandit):
    """A contextual bandit based on the Statlog (Shuttle) dataset.

    The dataset gives the recorded value of 9 different sensors during a space shuttle flight
    as well which state (out of 7 possible) the radiator was at each timestep.

    At each timestep the agent will get a 9-dimensional real valued context vector
    and must select one of 7 actions. The agent will get a reward of 1 only if it selects
    the true state of the radiator at that timestep as given in the dataset.

    Context dimension: 9
    Number of actions: 7

    Source:
        https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)

    Args:
        path (str, optional): Path to the data. Defaults to "./data/Statlog/".
        download (bool, optional): Whether to download the data. Defaults to False.
        force_download (bool, optional): Whether to force download even if file exists.
            Defaults to False.
        url (Union[str, None], optional): URL to download data from. Defaults to None
            which implies use of source URL.w
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
        super(StatlogDataBandit, self).__init__(kwargs.get("device", "cpu"))

        path = kwargs.get("path", "./data/Statlog/")
        download = kwargs.get("download", None)
        force_download = kwargs.get("force_download", None)
        url = kwargs.get("url", URL)

        if download:
            z_fpath = download_data(path, url, force_download)
            subprocess.run(["uncompress", "-f", z_fpath])
            fpath = Path(z_fpath).parent.joinpath("shuttle.trn")
            self.df = pd.read_csv(fpath, header=None, delimiter=" ")
        else:
            self.df = fetch_data_without_header(path, "shuttle.trn", delimiter=" ")

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
            self.df.iloc[self.idx, : self.context_dim],
            device=self.device,
            dtype=torch.float,
        )
