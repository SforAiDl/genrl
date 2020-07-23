from copy import deepcopy

from torch import optim as opt

from genrl.deep.agents.dqn.base import DQN
from genrl.deep.common import get_env_properties, get_model


class DuelingDQN(DQN):
    """Dueling DQN class

    Paper: https://arxiv.org/abs/1511.06581

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
        super(DuelingDQN, self).__init__(*args, create_model=False, **kwargs)

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, **kwargs) -> None:
        """Function to initialize Q-value model

        Initialises Q-value mode based on network type ["cnn", "mlp"]
        """
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "dueling")(
            input_dim, action_dim, self.layers
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, **kwargs)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)
