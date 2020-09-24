from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
import torch

from genrl.utils import set_seeds


class BaseAgent(ABC):
    """Base Agent Class

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        network: Any,
        env: Any,
        create_model: bool = True,
        batch_size: int = 64,
        gamma: float = 0.99,
        shared_layers=None,
        policy_layers: Tuple = (64, 64),
        value_layers: Tuple = (64, 64),
        lr_policy: float = 0.0001,
        lr_value: float = 0.001,
        **kwargs
    ):
        self.network = network
        self.env = env
        self.create_model = create_model
        self.batch_size = batch_size
        self.gamma = gamma
        self.shared_layers = shared_layers
        self.policy_layers = policy_layers
        self.rewards = []
        self.value_layers = value_layers
        self.lr_policy = lr_policy
        self.lr_value = lr_value

        self.seed = kwargs["seed"] if "seed" in kwargs else None
        self.render = kwargs["render"] if "render" in kwargs else False

        # Assign device
        device = kwargs["device"] if "device" in kwargs else "cpu"
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if self.seed is not None:
            set_seeds(self.seed, self.env)

    def _create_model(self) -> None:
        """Function to initialize all models of the agent"""
        raise NotImplementedError

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select action given state

        Action selection method

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        raise NotImplementedError

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
        """
        raise NotImplementedError

    def _load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`torch.tensor`): neural net weights
        """

        raise NotImplementedError

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        raise NotImplementedError

    def empty_logs(self):
        """Empties logs"""
        raise NotImplementedError
