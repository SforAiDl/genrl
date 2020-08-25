import torch

from genrl.agents.deep.dqn.base import DQN
from genrl.agents.deep.dqn.utils import ddqn_q_target


class DoubleDQN(DQN):
    """Double DQN Class

    Paper: https://arxiv.org/abs/1509.06461

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
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Get target Q values for the DQN

        Args:
            next_states (:obj:`torch.Tensor`): Next states for which target Q-values
                need to be found
            rewards (:obj:`list`): Rewards at each timestep for each environment
            dones (:obj:`list`): Game over status for each environment

        Returns:
            target_q_values (:obj:`torch.Tensor`): Target Q values for the DQN
        """
        return ddqn_q_target(self, next_states, rewards, dones)
