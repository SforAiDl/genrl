import pathlib

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
current_path = pathlib.Path(__file__).parent.absolute()


class DataBasedBandit(object):
    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_csv(path, header=None)
        self.idx = 0

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: int) -> int:
        raise NotImplementedError


class CovertypeDataBandit(DataBasedBandit):
    def __init__(
        self, path: pathlib.Path = current_path / "data/Covertype/covtype.data"
    ):
        super(CovertypeDataBandit, self).__init__(path)
        self.n_actions = len(self.df.iloc[:, -1].unique())
        self.context_dim = self.df.shape[1] - 1

    def reset(self) -> torch.Tensor:
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.idx = 0
        return self._get_context()

    def step(self, action: int) -> int:
        label = self.df.iloc[self.idx, self.context_dim]
        r = int(label == (action + 1))
        self.idx += 1
        return self._get_context(), r

    def _get_context(self) -> torch.Tensor:
        return torch.tensor(
            self.df.iloc[self.idx, : self.context_dim], device=device, dtype=dtype
        )
