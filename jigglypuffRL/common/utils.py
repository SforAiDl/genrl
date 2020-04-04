import torch


def save_params(algo, name=None, version=None):
    if name is None:
        name = algo.network_type

    checkpoint = {'algo': name,
                  'weights': algo.ac.state_dict()}

    if version is None:
        torch.save(checkpoint, "{}.pt".format(name))
    else:
        torch.save(checkpoint, "{}-{}.pt".format(name, version))
    return checkpoint


def load_params(name, version=None):
    try:
        if version is None:
            checkpoint = torch.load("{}.pt").format(name)
        else:
            checkpoint = torch.load("{}-{}.pt".format(name, version))
    except FileNotFoundError:
        raise Exception("Check name and version number again")
    return checkpoint
