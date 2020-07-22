from copy import deepcopy
from typing import Tuple

import numpy as np
from torch import optim as opt

from genrl.deep.agents.dqn.base import BaseDQN
from genrl.deep.agents.dqn.utils import get_projection_distribution
from genrl.deep.common import get_env_properties, get_model


class CategoricalDQN(BaseDQN):
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
        super(CategoricalDQN, self).__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.empty_logs()
        self.create_model()

    def create_model(self, *args) -> None:
        """Function to initialize Q-value model

        Initialises Q-value mode based on network type ["cnn", "mlp"]
        """
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "categorical")(
            input_dim, action_dim, self.layers, self.noisy_layers, self.num_atoms
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

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
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.reshape(-1, *self.env.obs_shape)
        actions = actions.reshape(-1, *self.env.action_shape).long()
        rewards = torch.FloatTensor(rewards).reshape(-1)
        next_states = next_states.reshape(-1, *self.env.obs_shape)
        dones = torch.FloatTensor(dones).reshape(-1)

        projection_distribution = get_projection_distribution(
            self, next_states, rewards, dones
        )
        dist = self.model(states)
        actions = actions.unsqueeze(1).expand(-1, 1, self.num_atoms)
        dist = dist.gather(1, actions).squeeze(1)
        dist.data.clamp_(0.01, 0.99)

        loss = -(projection_distribution * dist.log()).sum(1).mean()
        self.logs["value_loss"].append(loss.item())
        return loss

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        for timestep in range(update_interval):
            loss = self.get_q_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.reset_noise()
            self.target_model.reset_noise()

            if timestep % update_interval == 0:
                self.update_target_model()
