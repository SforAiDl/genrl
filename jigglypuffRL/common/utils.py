import torch
import torch.nn as nn
import os


def get_model(type_, name_):
    if type_ == "ac":
        from jigglypuffRL.common.actor_critic import get_actor_critic_from_name

        return get_actor_critic_from_name(name_)
    elif type_ == "v":
        from jigglypuffRL.common.values import get_value_from_name

        return get_value_from_name(name_)
    elif type == "p":
        from jigglypuffRL.common.policies import get_policy_from_name

        return get_policy_from_name(name_)
    raise ValueError


def mlp(sizes):
    """
    generate MLP model given sizes of each layer
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = nn.ReLU if j < len(sizes) - 2 else nn.Identity
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def evaluate(algo, num_timesteps=1000):
    s = algo.env.reset()
    ep, ep_r, ep_t = 0, 0, 0
    total_r = 0

    print("\nEvaluating...")
    for t in range(num_timesteps):
        a = algo.select_action(s)
        s1, r, done, _ = algo.env.step(a)
        ep_r += r
        total_r += r
        ep_t += 1

        if done:
            ep += 1
            print("Ep: {}, reward: {}, t: {}".format(ep, ep_r, ep_t))
            s = algo.env.reset()
            ep_r, ep_t = 0, 0
        else:
            s = s1

    algo.env.close()
    print("Average Reward: {}".format(total_r / num_timesteps))


def save_params(algo, directory=None):
    if directory is None:
        directory = "checkpoints"

    if not os.path.exists(directory):
        os.makedirs(directory)

    if algo.save_version is None:
        torch.save(algo.checkpoint, "{}/{}.pt".format(directory, algo.save_name))
    else:
        torch.save(
            algo.checkpoint, "{}/{}-{}.pt".format(
                directory, algo.save_name, algo.save_version
            )
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
