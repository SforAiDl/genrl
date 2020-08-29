import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Union

import pandas as pd


def download_data(
    path: str, url: str, force: bool = False, filename: Union[str, None] = None
) -> str:
    """Download data to given location from given URL.

    Args:
        path (str): Location to download to.
        url (str): URL to download file from.
        force (bool, optional): Force download even if file exists. Defaults to False.
        filename (Union[str, None], optional): Name to save file under. Defaults to None
            which implies original filename is to be used.

    Returns:
        str: Path to downloaded file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = Path(url).name
    fpath = path.joinpath(filename)
    if fpath.is_file() and not force:
        return str(fpath)

    try_data_download(fpath, path, url)

    return str(fpath)


def try_data_download(fpath: Path, path: Path, url: str):
    try:
        print(f"Downloading {url} to {fpath.resolve()}")
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead.")
            print(f" Downloading {url} to {path}")
            urllib.request.urlretrieve(url, fpath)
        else:
            raise e


def fetch_data_without_header(
    path: Union[str, Path], fname: str, delimiter: str = ",", na_values: list = []
):
    if Path(path).is_dir():
        path = Path(path).joinpath(fname)
    if Path(path).is_file():
        df = pd.read_csv(
            path, header=None, delimiter=delimiter, na_values=na_values
        ).dropna()
    else:
        raise FileNotFoundError(f"File not found at location {path}, use download flag")
    return df


def fetch_data_with_header(
    path: Union[str, Path], fname: str, delimiter: str = ",", na_values: list = []
):
    if Path(path).is_dir():
        path = Path(path).joinpath(fname)
    if Path(path).is_file():
        df = pd.read_csv(path, delimiter=delimiter, na_values=na_values).dropna()
    else:
        raise FileNotFoundError(f"File not found at location {path}, use download flag")
    return df


def fetch_zipped_data_without_header(
    gz_fpath: str, delimiter: str = ",", na_values: list = []
):
    with gzip.open(gz_fpath, "rb") as f_in:
        fpath = Path(gz_fpath).parent.joinpath("covtype.data")
        with open(fpath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    df = pd.read_csv(
        fpath, header=None, delimiter=delimiter, na_values=na_values
    ).dropna()
    fpath.unlink()
    return df
