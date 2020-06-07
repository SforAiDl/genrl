import os
import gym
import random
import torch
import numpy as np
import torch.nn as nn
from ...environments import VecEnv
from typing import Tuple, Union, Any


def get_model(type_: str, name_: str) -> Union:
    """
    Utility to get the class of required function

    :param type_: "ac" for Actor Critic, "v" for Value, "p" for Policy
    :param name_: Name of the specific structure of model. \
Eg. "mlp" or "cnn"
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


def mlp(sizes: Tuple, sac: bool = False):
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


def cnn(
    channels: Tuple = (4, 16, 32),
    kernel_sizes: Tuple = (8, 4),
    strides: Tuple = (4, 2),
    in_size: int = 84,
) -> (Tuple):
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


def save_params(algo: Any, timestep: int) -> None:
    """
    Function to save all parameters of a given agent

    :param algo: The agent object
    :param timestep: The timestep during training at which model is being saved
    :type algo: Object
    :type timestep: int
    """
    algo_name = algo.__class__.__name__
    if isinstance(algo.env, VecEnv):
        env_name = algo.env.envs[0].unwrapped.spec.id
    else:
        env_name = algo.env.unwrapped.spec.id
    directory = algo.save_model
    path = "{}/{}_{}".format(directory, algo_name, env_name)

    if algo.run_num is not None:
        run_num = algo.run_num
    else:
        if not os.path.exists(path):
            os.makedirs(path)
            run_num = 0
        elif list(os.scandir(path)) == []:
            run_num = 0
        else:
            last_path = sorted(os.scandir(path), key=lambda d: d.stat().st_mtime)[
                -1
            ].path
            run_num = int(last_path[len(path) + 1 :].split("-")[0]) + 1
        algo.run_num = run_num

    torch.save(
        algo.get_hyperparams(), "{}/{}-log-{}.pt".format(path, run_num, timestep)
    )


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


def get_env_properties(env: Union[gym.Env, VecEnv]) -> (Tuple[int]):
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


def get_obs_action_shape(obs, action):
    """
    Get the shapes of observation and action

    :param obs: State space of environment
    :param action: Action
    :type obs: gym.Space
    :type action: np.array
    """
    if isinstance(obs, gym.spaces.Discrete):
        return 1, 1
    elif isinstance(obs, gym.spaces.Box):
        return obs.shape[0], int(np.prod(action.shape))
    else:
        raise NotImplementedError


def get_obs_shape(observation_space):
    """
    Get the shape of the observation.

    :param observation_space: Observation space
    :type observation_space: gym.spaces.Space
    :returns: The observation space's shape
    :rtype: (Tuple[int, ...])
    """
    if isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Discrete):
        return (1,)
    else:
        raise NotImplementedError()


def get_action_dim(action_space):
    """
    Get the dimension of the action space.
    :param action_space: Action space
    :type action_space: gym.spaces.Space
    :returns: Action space's shape
    :rtype: int
    """
    if isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, gym.spaces.Discrete):
        return 1
    else:
        raise NotImplementedError()
