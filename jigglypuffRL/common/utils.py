import torch
import torch.nn as nn
import os
import pickle


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
    state = algo.env.reset()
    episode, episode_reward, episode_t = 0, 0, 0
    total_reward = 0

    print("\nEvaluating...")
    for t in range(num_timesteps):
        action = algo.select_action(state)
        next_state, reward, done, _ = algo.env.step(action)
        episode_reward += reward
        total_reward += reward
        episode_t += 1

        if done:
            episode += 1
            print("Ep: {}, reward: {}, t: {}".format(
                episode, episode_reward, episode_t
            ))
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
    path = "{}/{}_{}".format(
        directory, algo_name, env_name
    )

    if timestep == algo.save_interval:
        if not os.path.exists(directory):
            os.makedirs(directory)
            log = {}
            run_num = 0
        else:
            f = open("{}/log.pkl".format(directory), "rb")
            log = pickle.load(f)
            f.close()
            run_num = log[path] + 1

        if not os.path.exists(path):
            os.makedirs(path)
    else:
        f = open("{}/log.pkl".format(directory), "rb")
        log = pickle.load(f)
        f.close()
        run_num = log[path]

    torch.save(algo.checkpoint, "{}/{}-log-{}.pt".format(
        path, run_num, timestep
    ))

    log[path] = run_num
    log[path+str(run_num)] = timestep

    f = open("{}/log.pkl".format(directory), "wb")
    pickle.dump(log, f)
    f.close()


def load_params(algo):
    algo_name = algo.__class__.__name__
    env_name = algo.env.unwrapped.spec.id
    directory = algo.save_model
    run_num = algo.pretrained
    path = "{}/{}_{}".format(
        directory, algo_name, env_name
    )

    f = open("log.pkl", "rb")
    log = pickle.load(f)
    f.close()

    if run_num is None:
        run_num = log[path]
    timestep = log[path+str(run_num)]

    try:
        algo.checkpoint = torch.load(
            "{}/{}-log-{}.pt".format(path, run_num, timestep)
        )
    except FileNotFoundError:
        raise Exception("File name seems to be invalid")
    except NotADirectoryError:
        raise Exception("Invalid directory path")
