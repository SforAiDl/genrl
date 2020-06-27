from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
import torch

from genrl.deep.common.utils import load_params, save_params, set_seeds


class BaseAgent(ABC):
    def __init__(self, network_type: str, env: Any, epochs: int = 100, **kwargs):
        self.network_type = network_type
        self.env = env
        self.epochs = epochs
        self.seed = kwargs.get("seed", None)
        self.render = kwargs.get("render", False)
        self.run_num = kwargs.get("run_num", None)
        self.save_model = kwargs.get("save_model", None)
        self.load_model = kwargs.get("load_model", None)
        self.save_interval = kwargs.get("save_interval", 50)
        self.observation_space = None
        self.action_space = None

        # Assign device
        device = kwargs.get("device", "cpu")
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if self.seed is not None:
            set_seeds(self.seed, self.env)

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


class OnPolicyAgent(BaseAgent):
    def __init__(
        self,
        network_type: str,
        env: Any,
        batch_size: int = 256,
        layers: Tuple = (64, 64),
        gamma: float = 0.99,
        lr_policy: float = 0.01,
        lr_value: float = 0.0005,
        epochs: int = 100,
        rollout_size: int = 2048,
        **kwargs
    ):
        super(OnPolicyAgent, self).__init__(network_type, env, epochs, **kwargs)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.layers = layers
        self.rollout_size = rollout_size
        self.load = load_params
        self.save = save_params

    def collect_rewards(self, dones):
        for i, done in enumerate(dones):
            if done:
                self.rewards.append(self.epoch_reward[i])
                self.epoch_reward[i] = 0

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
                old_log_probs.detach(),
            )

            state = next_state

            self.collect_rewards(dones)

        return values, dones
