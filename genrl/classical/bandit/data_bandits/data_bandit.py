import urllib.request
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def download_data(
    path: str, url: str, force: bool = False, filename: Union[str, None] = None
) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(url).name
    fpath = path.joinpath(filename)
    if fpath.is_file() and not force:
        return str(fpath)

    try:
        print(f"Downloading {url} to {fpath}")
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead.")
            print(f" Downloading {url} to {path}")
            urllib.request.urlretrieve(url, fpath)
        else:
            raise e

    return str(fpath)


class DataBasedBandit(object):
    def __init__(self):
        self.idx = 0

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError
