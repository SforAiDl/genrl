import gym
import torch
import torch.nn as nn

def mlp(sizes):
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
    print("Average Reward: {}".format(total_r/num_timesteps))
    

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
