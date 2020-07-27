import urllib.request
from pathlib import Path
from typing import Union


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

    return str(fpath)
