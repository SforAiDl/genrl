import os
import random

import torch
import numpy as np
import torch.nn as nn


def get_model(type_, name_):
    if type_ == "ac":
        from genrl.deep.common.actor_critic import get_actor_critic_from_name

        return get_actor_critic_from_name(name_)
    elif type_ == "v":
        from genrl.deep.common.values import get_value_from_name

        return get_value_from_name(name_)
    elif type_ == "p":
        from genrl.deep.common.policies import get_policy_from_name

        return get_policy_from_name(name_)
    raise ValueError


def mlp(sizes, sac=False):
    """
    generate MLP model given sizes of each layer
    """
    layers = []
    limit = len(sizes) if sac is False else len(sizes) - 1
    for j in range(limit - 1):
        act = nn.ReLU if j < limit - 2 else nn.Identity
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def evaluate(algo, num_timesteps=1000):
    state = algo.env.reset()
    episode, episode_reward, episode_t = 0, 0, 0
    total_reward = 0

    print("\nEvaluating...")
    for _ in range(num_timesteps):
        action = algo.select_action(state)
        next_state, reward, done, _ = algo.env.step(action.item())
        episode_reward += reward
        total_reward += reward
        episode_t += 1

        if done:
            episode += 1
            print(
                "Episode: {}, Reward: {}, Timestep: {}".format(
                    episode, episode_reward, episode_t
                )
            )
            state = algo.env.reset()
            episode_reward, episode_t = 0, 0
        else:
            state = next_state

    algo.env.close()
    print("Average Reward: {}".format(total_reward / num_timesteps))


def save_params(algo, timestep):
    algo_name = algo.__class__.__name__
    env_name = algo.env.unwrapped.spec.id
    directory = algo.save_model
    path = "{}/{}_{}".format(directory, algo_name, env_name)

    if algo.run_num is not None:
        run_num = algo.run_num
    else:
        if not os.path.exists(path):
            os.makedirs(path)
            run_num = 0
        else:
            last_path = sorted(os.scandir(path), key=lambda d: d.stat().st_mtime)[
                -1
            ].path
            run_num = int(last_path[len(path) + 1 :].split("-")[0]) + 1
        algo.run_num = run_num

    torch.save(algo.checkpoint, "{}/{}-log-{}.pt".format(path, run_num, timestep))


def load_params(algo):
    path = algo.pretrained

    try:
        algo.checkpoint = torch.load(path)
    except FileNotFoundError:
        raise Exception("Invalid file name")


def set_seeds(seed, env=None):
    """
    Sets seeds for reproducibility
    :param seed: (int) Seed Value
    :param env: (gym env) Optionally pass gym environment to set its seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
