import os
import random

import torch
import numpy as np
import torch.nn as nn
import gym


def get_model(function_type, function_name):
    """
    Utility to get the class of required function

    :param function_type: "ac" for Actor Critic, "v" for Value, "p" for Policy
    :param function_name: Name of the specific structure of model. \
Eg. "mlp" or "cnn"
    :type function_type: string
    :type function_name: string
    :returns: Required class. Eg. MlpActorCritic
    """
    if function_type == "ac":
        from genrl.deep.common.actor_critic import get_actor_critic_from_name

        return get_actor_critic_from_name(function_name)
    elif function_type == "v":
        from genrl.deep.common.values import get_value_from_name

        return get_value_from_name(function_name)
    elif function_type == "p":
        from genrl.deep.common.policies import get_policy_from_name

        return get_policy_from_name(function_name)
    raise ValueError


def mlp(sizes, sac=False):
    """
    Generates an MLP model given sizes of each layer

    :param sizes: Sizes of hidden layers
    :param sac: True if Soft Actor Critic is being used, else False
    :type sizes: tuple or list
    :type sac: bool
    :returns: Neural Network with fully-connected linear layers and \
activation layers
    """
    layers = []
    limit = len(sizes) if sac is False else len(sizes) - 1
    for j in range(limit - 1):
        act = nn.ReLU if j < limit - 2 else nn.Identity
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def cnn(channels=(4, 16, 32), kernel_sizes=(8, 4), strides=(4, 2), in_size=84):
    """
    Generates a CNN model given input dimensions, channels, kernel_sizes and \
strides

    :param channels: Input output channels before and after each convolution
    :param kernel_sizes: Kernel sizes for each convolution
    :param strides: Strides for each convolution
    :param in_size: Input dimensions (assuming square input)
    :type channels: tuple
    :type kernel_sizes: tuple
    :type strides: tuple
    :type in_size: int
    :returns: Convolutional Neural Network with convolutional layers and \
activation layers
    """
    cnn_layers = []
    output_size = in_size

    for i in range(len(channels) - 1):
        in_channels, out_channels = channels[i], channels[i + 1]
        kernel_size, stride = kernel_sizes[i], strides[i]
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        activation = nn.ReLU()
        cnn_layers += [conv, activation]
        output_size = (output_size - kernel_size) / stride + 1

    cnn_layers = nn.Sequential(*cnn_layers)
    output_size = int(out_channels * (output_size ** 2))
    return cnn_layers, output_size


def evaluate(algo, num_timesteps=1000):
    """
    Function to evaluate the performance of a given agent

    :param algo: The agent object
    :param num_timesteps: Number of timesteps to evaluate agent over
    :type algo: Object
    :type num_timesteps: int
    """
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
    """
    Function to save all parameters of a given agent

    :param algo: The agent object
    :param timestep: The timestep during training at which model is being saved
    :type algo: Object
    :type timestep: int
    """
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
            last_path = sorted(
                os.scandir(path), key=lambda d: d.stat().st_mtime
            )[-1].path
            run_num = int(last_path[len(path) + 1 :].split("-")[0]) + 1
        algo.run_num = run_num

    torch.save(algo.checkpoint, "{}/{}-log-{}.pt".format(path, run_num, timestep))


def load_params(algo):
    """
    Function load parameters for an algorithm from a given checkpoint file

    :param algo: The agent object
    :type algo: Object
    """
    path = algo.load_model

    try:
        algo.checkpoint = torch.load(path)
    except FileNotFoundError:
        raise Exception("Invalid file name")


def get_env_properties(env):
    """
    Finds important properties of environment

    :param env: Environment that the agent is interacting with
    :type env: Gym Environment

    :returns: State space dimensions, Action space dimensions, \
discreteness of action space and action limit (highest action value)
    :rtype: int, float, ...; int, float, ...; bool; int, float, ...
    """
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
        action_lim = None
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_lim = env.action_space.high[0]
        discrete = False
    else:
        raise NotImplementedError

    return state_dim, action_dim, discrete, action_lim


def set_seeds(seed, env=None):
    """
    Sets seeds for reproducibility

    :param seed: Seed Value
    :param env: Optionally pass gym environment to set its seed
    :type seed: int
    :type env: Gym Environment
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
