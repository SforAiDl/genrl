import torch


def save_params(algo):
    if algo.save_version is None:
        torch.save(algo.checkpoint, "{}.pt".format(algo.save_name))
    else:
        torch.save(
            algo.checkpoint, "{}-{}.pt".format(algo.save_name, algo.save_version)
        )


def load_params(algo):
    try:
        if algo.save_version is None:
            algo.checkpoint = torch.load("{}.pt").format(algo.save_name)
        else:
            algo.checkpoint = torch.load(
                "{}-{}.pt".format(algo.save_name, algo.save_version)
            )
    except FileNotFoundError:
        raise Exception("Check name and version number again")
