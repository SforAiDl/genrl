import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from genrl.deep.common.utils import set_seeds


class BaseAgent(ABC):
    def __init__(
        self,
        network_type: str,
        env: VecEnv,
        num_episodes: int = 100,
        steps_per_episode: int = 1000,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
    ):
        self.network_type = network_type
        self.env = env
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.seed = seed
        self.render = render
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save_interval = save_interval
        self.observation_space = None
        self.action_space = None

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

    @abstractmethod
    def _create_model(self) -> None:
        """
        Initialize all the policy networks in this method, and also \
        initialize the optimizers and the buffers. 
        """
        raise NotImplementedError()

    def _update_params_before_select_action(self, timestep: int) -> None:
        """
        Update any parameters before selecting action like epsilon for decaying epsilon greedy

        :param timestep: Timestep in the training process
        :type timestep: int
        """
        pass

    @abstractmethod
    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Selection of action

        :param state: Observation state
        :type state: int, float, ...
        :returns: Action based on the state and epsilon value
        :rtype: int, float, ...
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_loss(self) -> None:
        """
        Calculate and return the total loss
        """
        raise NotImplementedError()

    @abstractmethod
    def _update_params(self, update_interval: int) -> None:
        """
        Takes the step for optimizer. 
        This internally call _get_loss().
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """
        Hyperparameters you want to return 
        """
        raise NotImplementedError()
