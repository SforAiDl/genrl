from pathlib import Path


def write_data(fname, d):
    dpath = Path("./data")
    dpath.mkdir(exist_ok=True)
    fpath = dpath.joinpath(fname)
    with open(fpath, "w") as f:
        f.write(d)
    return fpath
