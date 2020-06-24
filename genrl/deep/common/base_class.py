from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from genrl.deep.common.utils import set_seeds


class BaseAgent(ABC):
    def __init__(
        self,
        network_type: str,
        env: Any,
        epochs: int = 100,
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
        self.epochs = epochs
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

    def create_model(self) -> None:
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Selection of action

        :param state: Observation state
        :type state: int, float, ...
        :returns: Action based on the state and epsilon value
        :rtype: int, float, ...
        """
        raise NotImplementedError()

    def get_loss(self) -> None:
        """
        Calculate and return the total loss
        """
        raise NotImplementedError()

    def update_params(self, update_interval: int) -> None:
        """
        Takes the step for optimizer.
        This internally call _get_loss().
        """
        raise NotImplementedError()

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Hyperparameters you want to return
        """
        raise NotImplementedError()


class BaseOnPolicyAgent(BaseAgent):
    def __init__(
        self,
        network_type: str,
        env: Any,
        timesteps_per_actorbatch: int = 256,
        layers: Tuple = (64, 64),
        gamma: float = 0.99,
        lr_policy: float = 0.01,
        lr_value: float = 0.0005,
        actor_batch_size: int = 64,
        epochs: int = 100,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
        rollout_size: int = 2048,
    ):
        super(BaseOnPolicyAgent, self).__init__(
            network_type,
            env,
            epochs,
            seed,
            render,
            device,
            run_num,
            save_model,
            load_model,
            save_interval,
        )
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.gamma = gamma
        self.actor_batch_size = actor_batch_size
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.layers = layers
        self.rollout_size = rollout_size

    def collect_rollouts(self, initial_state):

        state = initial_state

        for i in range(self.rollout_size):
            # with torch.no_grad():
            action, values, old_log_probs = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(np.array(action))
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                dones,
                values.detach(),
                old_log_probs,
            )

            state = next_state

            for i, done in enumerate(dones):
                if done:
                    self.rewards.append(self.epoch_reward[i])
                    self.epoch_reward[i] = 0

        return values, done
