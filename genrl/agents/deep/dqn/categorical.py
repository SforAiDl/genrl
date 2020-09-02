import collections
from typing import Tuple

import torch

from genrl.agents.deep.dqn.base import DQN
from genrl.agents.deep.dqn.utils import (
    categorical_greedy_action,
    categorical_q_loss,
    categorical_q_target,
    categorical_q_values,
)


class CategoricalDQN(DQN):
    """Categorical DQN Algorithm

    Paper: https://arxiv.org/pdf/1707.06887.pdf

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
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
        num_atoms (int): Number of atoms used in the discrete distribution
        v_min (int): Lower bound of value distribution
        v_max (int): Upper bound of value distribution
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        noisy_layers: Tuple = (32, 128),
        num_atoms: int = 51,
        v_min: int = -10,
        v_max: int = 10,
        **kwargs
    ):
        super(CategoricalDQN, self).__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.dqn_type = "categorical"
        self.noisy = True

        self.empty_logs()
        if self.create_model:
            self._create_model(noisy_layers=self.noisy_layers, num_atoms=self.num_atoms)

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy action selection

        Args:
            state (:obj:`torch.Tensor`): Current state of the environment

        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        return categorical_greedy_action(self, state)

    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor):
        """Get Q values corresponding to specific states and actions

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states

        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        return categorical_q_values(self, states, actions)

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ):
        """Projected Distribution of Q-values

        Helper function for Categorical/Distributional DQN

        Args:
            next_states (:obj:`torch.Tensor`): Next states being encountered by the agent
            rewards (:obj:`torch.Tensor`): Rewards received by the agent
            dones (:obj:`torch.Tensor`): Game over status of each environment

        Returns:
            target_q_values (object): Projected Q-value Distribution or Target Q Values
        """
        return categorical_q_target(self, next_states, rewards, dones)

    def get_q_loss(self, batch: collections.namedtuple):
        """Categorical DQN loss function to calculate the loss of the Q-function

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        return categorical_q_loss(self, batch)
