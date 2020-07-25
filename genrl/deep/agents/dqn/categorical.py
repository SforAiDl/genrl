from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import optim as opt

from genrl.deep.agents.dqn.base import DQN
from genrl.deep.agents.dqn.utils import get_projection_distribution
from genrl.deep.common import get_env_properties, get_model


class CategoricalDQN(DQN):
    """Categorical DQN Algorithm

    Paper: https://arxiv.org/pdf/1707.06887.pdf

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
        self.noisy_layers = noisy_layers
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.dqn_type = "categorical"
        self.noisy = True

        super(CategoricalDQN, self).__init__(*args, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model(noisy_layers=self.noisy_layers, num_atoms=self.num_atoms)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action given state

        Epsilon-greedy action-selection

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        if not deterministic:
            if np.random.rand() < self.epsilon:
                return np.asarray(self.env.sample())

        state = torch.FloatTensor(state)
        dist = self.model(state).data.cpu()
        dist = dist * torch.linspace(self.v_min, self.v_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()
        return action

    def get_q_loss(self):
        """Function to calculate the loss of the Q-function

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        batch = self.sample_from_buffer()

        projection_distribution = get_projection_distribution(
            self, batch.next_states, batch.rewards, batch.dones
        )
        dist = self.model(batch.states)
        actions = batch.actions.unsqueeze(1).expand(-1, 1, self.num_atoms)
        dist = dist.gather(1, actions).squeeze(1)
        dist.data.clamp_(0.01, 0.99)

        loss = -(projection_distribution * dist.log()).sum(1).mean()
        self.logs["value_loss"].append(loss.item())
        return loss
