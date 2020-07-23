import torch

from genrl.deep.agents.dqn.base import DQN


class DoubleDQN(DQN):
    """Double DQN Class

    Paper: https://arxiv.org/abs/1509.06461

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
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, create_model=False, **kwargs)
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
        next_q_values = self.model(next_states)
        next_best_actions = next_q_values.max(1)[1].unsqueeze(1)

        rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

        next_q_target_values = self.target_model(next_states)
        max_next_q_target_values = next_q_target_values.gather(1, next_best_actions)
        target_q_values = rewards + self.gamma * torch.mul(
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values
