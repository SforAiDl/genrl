from copy import deepcopy
from typing import Tuple

from torch import optim as opt

from genrl.deep.agents.dqn.base import BaseDQN
from genrl.deep.common import get_env_properties, get_model


class NoisyDQN(BaseDQN):
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
        super(NoisyDQN, self).__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers

        self.empty_logs()
        self.create_model()

    def create_model(self, *args) -> None:
        """Function to initialize Q-value model

        Initialises Q-value mode based on network type ["cnn", "mlp"]
        """
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "noisy")(
            input_dim, action_dim, fc_layers=self.layers, noisy_layers=self.noisy_layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

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
