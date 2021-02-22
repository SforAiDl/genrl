import collections
from abc import ABC

import torch
import torch.nn as nn
import torch.optim as opt

from genrl.core import MultiAgentReplayBuffer
from genrl.utils import MutiAgentEnvInterface


class MultiAgentOffPolicy(ABC):
    """Base class for multiagent algorithms with OffPolicy agents

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        agents (list) : A list of all the agents to be used
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        replay_size (int): Capacity of the Replay Buffer
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    raise NotImplementedError
