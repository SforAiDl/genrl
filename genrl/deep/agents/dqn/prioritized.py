import torch
from torch import optim as opt

from genrl.deep.agents.dqn.base import DQN


class PrioritizedReplayDQN(DQN):
    """Prioritized Replay DQN Class

    Paper: https://arxiv.org/abs/1511.05952

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
        alpha (float): Prioritization constant
        beta (float): Importance Sampling bias
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, alpha: float = 0.6, beta: float = 0.4, **kwargs):
        self.alpha = alpha
        self.beta = beta

        super(PrioritizedReplayDQN, self).__init__(
            *args, buffer_type="prioritized", **kwargs
        )

        self.empty_logs()
        if self.create_model:
            self._create_model(alpha=self.alpha)

    def get_q_loss(self) -> torch.Tensor:
        """Function to calculate the loss of the Q-function

        Returns:
            loss (:obj:`torch.Tensor`): Calculateed loss of the Q-function
        """
        batch = self.sample_from_buffer(beta=self.beta)

        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )

        loss = batch.weights * (q_values - target_q_values.detach()) ** 2
        priorities = loss + 1e-5
        loss = loss.mean()
        self.replay_buffer.update_priorities(
            batch.indices, priorities.detach().cpu().numpy()
        )
        self.logs["value_loss"].append(loss.item())
        return loss
