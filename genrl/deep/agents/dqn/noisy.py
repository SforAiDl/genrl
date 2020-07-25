from copy import deepcopy
from typing import Tuple

from torch import optim as opt

from genrl.deep.agents.dqn.base import DQN
from genrl.deep.common import get_env_properties, get_model


class NoisyDQN(DQN):
    """Noisy DQN Algorithm

    Paper: https://arxiv.org/abs/1706.10295

    Attributes:
        network_type (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_value (float): Learning rate for the Q-value function
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        max_epsilon (str): Maximum epsilon for exploration
        min_epsilon (str): Minimum epsilon for exploration
        epsilon_decay (str): Rate of decay of epsilon (in order to decrease
            exploration with time)
        noisy_layers (:obj:`tuple` of :obj:`int`): Noisy layers in the Neural
            Network of the Q-value function
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, noisy_layers: Tuple = (128, 128), **kwargs):
        self.noisy_layers = noisy_layers
        self.noisy = True
        super(NoisyDQN, self).__init__(*args, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model(noisy_layers=self.noisy_layers)
