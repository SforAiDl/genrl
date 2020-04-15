import torch
import torch.nn as nn


def get_model(type_, name_):
    if type_ == "ac":
        from jigglypuffRL.common.actor_critic import get_actor_critic_from_name

        return get_actor_critic_from_name(name_)
    elif type_ == "v":
        from jigglypuffRL.common.values import get_value_from_name

        return get_value_from_name(name_)
    elif type_ == "p":
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
    state = algo.env.reset()
    episode, episode_reward, episode_t = 0, 0, 0
    total_reward = 0

    for t in range(num_timesteps):
        if algo.render:
            algo.env.render()
        action = algo.select_action(state)
        next_state, reward, done, _ = algo.env.step(action)
        episode_reward += reward
        episode_t += 1

        if done:
            episode += 1
            print("Ep: {}, reward: {}, t: {}".format(
                episode, episode_reward, episode_t
            ))
            state = algo.env.reset()
            total_reward += episode_reward
            episode_reward, episode_t = 0, 0
        else:
            state = next_state

    algo.env.close()
    print("Average Reward: {}".format(total_reward / episode))


def save_params(algo):
    if algo.save_version is None:
        torch.save(algo.checkpoint, "{}.pt".format(algo.save_name))
    else:
        torch.save(
            algo.checkpoint, "{}-{}.pt".format(
                algo.save_name, algo.save_version
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
