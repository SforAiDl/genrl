import collections

import torch

from genrl.agents.deep.dqn.base import DQN
from genrl.agents.deep.dqn.utils import prioritized_q_loss


class PrioritizedReplayDQN(DQN):
    """Prioritized Replay DQN Class

    Paper: https://arxiv.org/abs/1511.05952

    Attributes:
        network (str): The network type of the Q-value function.
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
        alpha (float): Prioritization constant
        beta (float): Importance Sampling bias
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        super(PrioritizedReplayDQN, self).__init__(
            *args, buffer_type="prioritized", **kwargs
        )
        self.replay_buffer.alpha = alpha
        self.replay_buffer.beta = beta

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Normal Function to calculate the loss of the Q-function

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        return prioritized_q_loss(self, batch)
