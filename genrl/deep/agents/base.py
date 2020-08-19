import collections
from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from genrl.deep.common.actor_critic import BaseActorCritic
from genrl.deep.common.buffers import (
    PrioritizedBuffer,
    PrioritizedReplayBufferSamples,
    PushReplayBuffer,
    ReplayBufferSamples,
)
from genrl.deep.common.utils import set_seeds


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
        network: Union[str, BaseActorCritic],
        env: Any,
        create_model: bool = True,
        batch_size: int = 64,
        gamma: float = 0.99,
        policy_layers: Tuple = (64, 64),
        value_layers: Tuple = (64, 64),
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        **kwargs
    ):
        self.network = network
        self.env = env
        self.create_model = create_model
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_layers = policy_layers
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
        """Function to initialize all models of the agent
        """
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

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        raise NotImplementedError

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
        """
        raise NotImplementedError

    def get_logging_params(self) -> Dict[str, Any]:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`dict`): Dictionary of different neural net weights
        """
        raise NotImplementedError

    def empty_logs(self):
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
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
    """Off Policy Agent Base Class

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
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self, *args, replay_size: int = 1000, buffer_type: str = "push", **kwargs
    ):
        super(OffPolicyAgent, self).__init__(*args, **kwargs)
        self.replay_size = replay_size
        self.buffer_type = buffer_type

        if buffer_type == "push":
            self.buffer_class = PushReplayBuffer
        elif buffer_type == "prioritized":
            self.buffer_class = PrioritizedBuffer
        else:
            raise NotImplementedError

    def update_params_before_select_action(self, timestep: int) -> None:
        """Update any parameters before selecting action like epsilon for decaying epsilon greedy

        Args:
            timestep (int): Timestep in the training process
        """
        pass

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        raise NotImplementedError

    def _reshape_batch(self, batch: List):
        """Function to reshape experiences

        Can be modified for individual algorithm usage

        Args:
            batch (:obj:`list`): List of experiences that are being replayed

        Returns:
            batch (:obj:`list`): Reshaped experiences for replay
        """
        return [*batch]

    def sample_from_buffer(self, beta: float = None):
        """Samples experiences from the buffer and converts them into usable formats

        Args:
            beta (float): Importance-Sampling beta for prioritized replay

        Returns:
            batch (:obj:`list`): Replay experiences sampled from the buffer
        """
        # Samples from the buffer
        if beta is not None:
            batch = self.replay_buffer.sample(self.batch_size, beta=beta)
        else:
            batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = self._reshape_batch(batch)

        # Convert every experience to a Named Tuple. Either Replay or Prioritized Replay samples.
        if self.buffer_type == "push":
            batch = ReplayBufferSamples(*[states, actions, rewards, next_states, dones])
        elif self.buffer_type == "prioritized":
            indices, weights = batch[5], batch[6]
            batch = PrioritizedReplayBufferSamples(
                *[states, actions, rewards, next_states, dones, indices, weights]
            )
        return batch

    def get_q_loss(self, batch: collections.namedtuple) -> torch.Tensor:
        """Normal Function to calculate the loss of the Q-function or critic

        Args:
            batch (:obj:`collections.namedtuple` of :obj:`torch.Tensor`): Batch of experiences

        Returns:
            loss (:obj:`torch.Tensor`): Calculated loss of the Q-function
        """
        q_values = self.get_q_values(batch.states, batch.actions)
        target_q_values = self.get_target_q_values(
            batch.next_states, batch.rewards, batch.dones
        )
        loss = F.mse_loss(q_values, target_q_values)
        return loss


class OffPolicyAgentAC(OffPolicyAgent):
    """Off Policy Agent Base Class

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
        replay_size (int): Capacity of the Replay Buffer
        buffer_type (str): Choose the type of Buffer: ["push", "prioritized"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(self, *args, polyak=0.995, **kwargs):
        super(OffPolicyAgentAC, self).__init__(*args, **kwargs)
        self.polyak = polyak

    def select_action(
        self, state: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """Select action given state

        Deterministic Action Selection with Noise

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`np.ndarray`): Action taken by the agent
        """
        state = torch.as_tensor(state).float()
        action, _ = self.ac.get_action(state, deterministic)
        action = action.detach().cpu().numpy()

        # add noise to output from policy network
        if self.noise is not None:
            action += self.noise()

        return np.clip(
            action, self.env.action_space.low[0], self.env.action_space.high[0]
        )

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        for param, param_target in zip(
            self.ac.parameters(), self.ac_target.parameters()
        ):
            param_target.data.mul_(self.polyak)
            param_target.data.add_((1 - self.polyak) * param.data)

    def get_p_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Function to get the Policy loss

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found

        Returns:
            loss (:obj:`torch.Tensor`): Calculated policy loss
        """
        next_best_actions = self.ac.get_action(states, True)[0]
        q_values = self.ac.get_value(torch.cat([states, next_best_actions], dim=-1))
        policy_loss = -torch.mean(q_values)
        return policy_loss

    def load_weights(self, weights) -> None:
        """
        Load weights for the agent from pretrained model
        """
        self.ac.load_state_dict(weights["weights"])
