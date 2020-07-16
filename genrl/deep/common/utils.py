import random
from typing import Any, List, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from ...environments import VecEnv


def get_model(type_: str, name_: str) -> Union:
    """
    Utility to get the class of required function

    :param type_: "ac" for Actor Critic, "v" for Value, "p" for Policy
    :param name_: Name of the specific structure of model. (
Eg. "mlp" or "cnn")
    :type type_: string
    :type name_: string
    :returns: Required class. Eg. MlpActorCritic
    """
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


def mlp(sizes: Tuple, sac: bool = False, activation: str = "relu"):
    """
    Generates an MLP model given sizes of each layer

    :param sizes: Sizes of hidden layers
    :param sac: True if Soft Actor Critic is being used, else False
    :type sizes: tuple or list
    :type sac: bool
    :returns: (Neural Network with fully-connected linear layers and
activation layers)
    """
    layers = []
    limit = len(sizes) if sac is False else len(sizes) - 1

    activation = nn.Tanh() if activation == "tanh" else nn.ReLU()

    for layer in range(limit - 1):
        act = activation if layer < limit - 2 else nn.Identity()
        layers += [nn.Linear(sizes[layer], sizes[layer + 1]), act]

    return nn.Sequential(*layers)


def cnn(
    channels: Tuple = (4, 16, 32),
    kernel_sizes: Tuple = (8, 4),
    strides: Tuple = (4, 2),
    **kwargs,
) -> (Tuple):
    """
    (Generates a CNN model given input dimensions, channels, kernel_sizes and
strides)

    :param channels: Input output channels before and after each convolution
    :param kernel_sizes: Kernel sizes for each convolution
    :param strides: Strides for each convolution
    :param in_size: Input dimensions (assuming square input)
    :type channels: tuple
    :type kernel_sizes: tuple
    :type strides: tuple
    :type in_size: int
    :returns: (Convolutional Neural Network with convolutional layers and
activation layers)
    """

    cnn_layers = []
    output_size = kwargs["in_size"] if "in_size" in kwargs else 84

    act_fn = kwargs["activation"] if "activation" in kwargs else "relu"
    activation = nn.Tanh() if act_fn == "tanh" else nn.ReLU()

    for i in range(len(channels) - 1):
        in_channels, out_channels = channels[i], channels[i + 1]
        kernel_size, stride = kernel_sizes[i], strides[i]
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        cnn_layers += [conv, activation]
        output_size = (output_size - kernel_size) / stride + 1

    cnn_layers = nn.Sequential(*cnn_layers)
    output_size = int(out_channels * (output_size ** 2))
    return cnn_layers, output_size


def load_params(algo: Any) -> None:
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


def get_env_properties(
    env: Union[gym.Env, VecEnv], network_type: str = "mlp"
) -> (Tuple[int]):
    """
    Finds important properties of environment

    :param env: Environment that the agent is interacting with
    :type env: Gym Environment
    :param network_type: Type of network architecture, eg. "mlp", "cnn"
    :type network_type: str
    :returns: (State space dimensions, Action space dimensions,
discreteness of action space and action limit (highest action value)
    :rtype: int, float, ...; int, float, ...; bool; int, float, ...
    """
    if network_type == "cnn":
        input_dim = env.framestack
    elif network_type == "mlp":
        input_dim = env.observation_space.shape[0]

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

    return input_dim, action_dim, discrete, action_lim


def set_seeds(seed: int, env: Union[gym.Env, VecEnv] = None) -> None:
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


def safe_mean(log: List[int]):
    """
    Returns 0 if there are no elements in logs
    """
    return np.mean(log) if len(log) > 0 else 0
