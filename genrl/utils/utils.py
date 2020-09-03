import random
from typing import Any, List, Tuple, Union

import gym
import numpy as np
import torch  # noqa
import torch.nn as nn  # noqa

from genrl.core.base import BaseActorCritic, BasePolicy, BaseValue
from genrl.core.noise import NoisyLinear
from genrl.environments.vec_env import VecEnv


def get_model(type_: str, name_: str) -> Union:
    """
        Utility to get the class of required function

        :param type_: "ac" for Actor Critic, "v" for Value, "p" for Policy
        :param name_: Name of the specific structure of model. (
    Eg. "mlp" or "cnn")
        :type type_: string
        :returns: Required class. Eg. MlpActorCritic
    """
    if type_ == "ac":
        from genrl.core import get_actor_critic_from_name

        return get_actor_critic_from_name(name_)
    elif type_ == "v":
        from genrl.core import get_value_from_name

        return get_value_from_name(name_)
    elif type_ == "p":
        from genrl.core import get_policy_from_name

        return get_policy_from_name(name_)
    raise ValueError


def mlp(
    sizes: Tuple,
    activation: str = "relu",
    sac: bool = False,
):
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


# If at all you need to concatenate states to actions after passing states through n FC layers
def mlp_(
    self,
    layer_sizes,
    weight_init,
    activation_func,
    concat_ind,
    sac
    ):
    """
        Generates an MLP model given sizes of each layer

        :param layer_sizes: Sizes of hidden layers
        :param weight_init: type of weight initialization
        :param activation_func: type of activation function
        :param concat_ind: index of layer at which actions to be concatenated
        :param sac: True if Soft Actor Critic is being used, else False
        :type layer_sizes: tuple or list
        :type concat_ind: int
        :type sac: bool
        :type weight_init,activation_func: string
        :returns: (Neural Network with fully-connected linear layers and
    activation layers)
    """
    layers = []
    limit = len(layer_sizes) if sac is False else len(sizes) - 1

    # add more activations
    activation = nn.Tanh() if activation_func == "tanh" else nn.ReLU()

    # add more weight init
    if weight_init == "xavier_uniform":
        weight_init = torch.nn.init.xavier_uniform_
    elif weight_init == "xavier_normal":
        weight_init = torch.nn.init.xavier_normal_


    for layer in range(limit - 1):
        if layer==concat_ind:
            continue
        act = activation if layer < limit - 2 else nn.Identity()
        layers += [nn.Linear(sizes[layer], sizes[layer + 1]), act]
        weight_init(layers[-1][0].weight)


def shared_mlp(
    network1_prev,
    network2_prev,
    shared,
    network1_post,
    network2_post,
    weight_init,
    activation_func,
    sac
    )
"""
        Generates an MLP model given sizes of each layer (Mostly used for SharedActorCritic)

        :param network1_prev: Sizes of network1's initial layers
        :param network2_prev: Sizes of network2's initial layers
        :param shared: Sizes of shared layers
        :param network1_post: Sizes of network1's latter layers
        :param network2_post: Sizes of network2's latter layers
        :param weight_init: type of weight initialization
        :param activation_func: type of activation function
        :param sac: True if Soft Actor Critic is being used, else False
        :type network1_prev,network2_prev,shared,network1_post,network2_post: tuple or list
        :type weight_init,activation_func: string
        :type sac: bool
        :returns: network1 and networ2(Neural Network with fully-connected linear layers and
    activation layers)
    """

    if len(network1_prev) != 0:
        network1_prev = nn.ModuleList()
    if len(network2_prev) != 0:
        network2_prev = nn.ModuleList()
    if len(shared) != 0:
        shared = nn.ModuleList()
    if len(network1_post) != 0:
        network1_post = nn.ModuleList()
    if len(network2_post) != 0:
        network2_post = nn.ModuleList()


    # add more weight init
    if weight_init == "xavier_uniform":
        weight_init = torch.nn.init.xavier_uniform_
    elif weight_init == "xavier_normal":
        weight_init = torch.nn.init.xavier_normal_
    else:
        weight_init = None

    if activation_func == "relu":
            activation = nn.ReLU()
        elif activation_func == "tanh":
            activation = nn.Tanh()
        else:
            activation = None

    if len(shared) != 0 or len(network1_post) != 0 or len(network2_post) != 0:
        if not (network1_prev[-1]==network2_prev[-1] and network1_prev[-1]==shared[0] and network1_post[0]==network2_post[0] and network1_post[0]==shared[-1]):
            raise ValueError

    for i in range(len(network1_prev)-1):
        network1_prev.append(nn.Linear(network1_prev[i],network1_prev[i+1]))
        if activation is not None:
            network1_prev.append(activation)
        if weight_init is not None:
            weight_init(network1_prev[-1].weight)

    for i in range(len(network2_prev)-1):
        network2_prev.append(nn.Linear(network2_prev[i],network2_prev[i+1]))
        if activation is not None:
            network2_prev.append(activation)
        if weight_init is not None:
            weight_init(network2_prev[-1].weight)

    for i in range(len(shared)-1):
        shared.append(nn.Linear(shared[i], shared[i+1]))
        if activation is not None:
            shared.append(activation)
        if weight_init is not None:
            weight_init(shared[-1].weight)

    for i in range(len(network1_post)-1):
        network1_post.append(nn.Linear(network1_post[i],network1_post[i+1]))
        if activation is not None:
            network1_post.append(activation)
        if weight_init is not None:
            weight_init(network1_post[-1].weight)

    for i in range(len(network2_post)-1):
        network2_post.append(nn.Linear(network2_post[i],network2_post[i+1]))
        if activation is not None:
            network2_post.append(activation)
        if weight_init is not None:
            weight_init(network2_post[-1].weight)


    network1 = nn.Sequential(network1_prev,shared,network1_post)
    network2 = nn.Sequential(network2_prev,shared,network2_post)

    return network1,network2


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


def noisy_mlp(fc_layers: List[int], noisy_layers: List[int], activation="relu"):
    """Noisy MLP generating helper function

    Args:
        fc_layers (:obj:`list` of :obj:`int`): List of fully connected layers
        noisy_layers (:obj:`list` of :obj:`int`): :ist of noisy layers
        activation (str): Activation function to be used. ["tanh", "relu"]

    Returns:
        Noisy MLP model
    """
    model = []
    act = nn.Tanh if activation == "tanh" else nn.ReLU()

    for layer in range(len(fc_layers) - 1):
        model += [nn.Linear(fc_layers[layer], fc_layers[layer + 1]), act]

    model += [nn.Linear(fc_layers[-1], noisy_layers[0]), act]

    for layer in range(len(noisy_layers) - 1):
        model += [NoisyLinear(noisy_layers[layer], noisy_layers[layer + 1])]
        if layer < len(noisy_layers) - 2:
            model += [act]

    return nn.Sequential(*model)


def get_env_properties(
    env: Union[gym.Env, VecEnv], network: Union[str, Any] = "mlp"
) -> (Tuple[int]):
    """
        Finds important properties of environment

        :param env: Environment that the agent is interacting with
        :type env: Gym Environment
        :param network: Type of network architecture, eg. "mlp", "cnn"
        :type network: str
        :returns: (State space dimensions, Action space dimensions,
    discreteness of action space and action limit (highest action value)
        :rtype: int, float, ...; int, float, ...; bool; int, float, ...
    """
    if network == "cnn":
        state_dim = env.framestack
    elif network == "mlp":
        state_dim = env.observation_space.shape[0]
    elif isinstance(network, (BasePolicy, BaseValue)):
        state_dim = network.state_dim
    elif isinstance(network, BaseActorCritic):
        state_dim = network.actor.state_dim
    else:
        raise TypeError

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


def safe_mean(log: Union[torch.Tensor, List[int]]):
    """
    Returns 0 if there are no elements in logs
    """

    if len(log) == 0:
        return 0
    if isinstance(log, torch.Tensor):
        func = torch.mean
    else:
        func = np.mean
    return func(log)
