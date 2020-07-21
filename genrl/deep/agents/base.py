from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
import torch

from genrl.deep.common.buffers import PrioritizedBuffer, PushReplayBuffer
from genrl.deep.common.utils import set_seeds


class BaseAgent(ABC):
    def __init__(
        self,
        network_type: str,
        env: Any,
        batch_size: int = 64,
        gamma: float = 0.99,
        layers: Tuple = (64, 64),
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        **kwargs
    ):
        self.network_type = network_type
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.layers = layers
        self.lr_policy = lr_policy
        self.lr_value = lr_value

        self.seed = kwargs["seed"] if "seed" in kwargs else None
        self.render = kwargs["render"] if "render" in kwargs else False

        # Assign device
        device = kwargs[device] if "device" in kwargs else "cpu"
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
        raise NotImplementedError

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Selection of action

        :param state: Observation state
        :type state: int, float, ...
        :returns: Action based on the state and epsilon value
        :rtype: int, float, ...
        """
        raise NotImplementedError

    def update_params(self, update_interval: int) -> None:
        """
        Takes the step for optimizer.
        This internally call _get_loss().
        """
        raise NotImplementedError

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Hyperparameters you want to return
        """
        raise NotImplementedError

    def get_logging_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def empty_logs(self):
        raise NotImplementedError


class OnPolicyAgent(BaseAgent):
    def __init__(self, *args, rollout_size: int = 2048, **kwargs):
        super(OnPolicyAgent, self).__init__(*args, **kwargs)
        self.rollout_size = rollout_size

    def collect_rewards(self, dones, timestep):
        for i, done in enumerate(dones):
            if done or timestep == self.rollout_size - 1:
                self.rewards.append(self.env.episode_reward[i])
                self.env.episode_reward[i] = 0

    def collect_rollouts(self, state):
        for i in range(self.rollout_size):
            action, values, old_log_probs = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(np.array(action))

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

            self.collect_rewards(dones, i)

        return values, dones


class OffPolicyAgent(BaseAgent):
    def __init__(self, *args, replay_size=1000000, buffer_type="push", **kwargs):
        super(OffPolicyAgent, self).__init__(*args, **kwargs)
        self.replay_size = replay_size

        if buffer_type == "push":
            self.buffer_class = PushReplayBuffer
        elif buffer_type == "prioritized":
            self.buffer_class = PrioritizedBuffer
        else:
            raise NotImplementedError
