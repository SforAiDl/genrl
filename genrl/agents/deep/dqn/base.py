import math
import random
from copy import deepcopy
from typing import Any, Dict, List

import torch  # noqa
import torch.optim as opt  # noqa

from genrl.agents import OffPolicyAgent
from genrl.utils import get_env_properties, get_model, safe_mean


class DQN(OffPolicyAgent):
    """Base DQN Class

    Paper: https://arxiv.org/abs/1312.5602

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        value_layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
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

    def __init__(
        self,
        *args,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 500,
        **kwargs
    ):
        super(DQN, self).__init__(*args, **kwargs)
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.dqn_type = ""
        self.noisy = False

        self.empty_logs()
        if self.create_model:
            self._create_model()

    def _create_model(self, *args, **kwargs) -> None:
        """Function to initialize Q-value model

        This will create the Q-value function of the agent.
        """
        state_dim, action_dim, discrete, _ = get_env_properties(self.env, self.network)
        if not discrete:
            raise Exception("Only Discrete Environments are supported for DQN")

        if isinstance(self.network, str):
            self.model = get_model("v", self.network + self.dqn_type)(
                state_dim, action_dim, "Qs", self.value_layers, **kwargs
            )
        else:
            self.model = self.network

        self.target_model = deepcopy(self.model)

        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

    def update_target_model(self) -> None:
        """Function to update the target Q model

        Updates the target model with the training model's weights when called
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def update_params_before_select_action(self, timestep: int) -> None:
        """Update necessary parameters before selecting an action

        This updates the epsilon (exploration rate) of the agent every timestep

        Args:
            timestep (int): Timestep of training
        """
        self.timestep = timestep
        self.epsilon = self.calculate_epsilon_by_frame()
        self.logs["epsilon"].append(self.epsilon)

    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy action selection

        Args:
            state (:obj:`torch.Tensor`): Current state of the environment

        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        q_values = self.model(state.unsqueeze(0))
        action = torch.argmax(q_values.squeeze(), dim=-1)
        return action

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Select action given state

        Epsilon-greedy action-selection

        Args:
            state (:obj:`torch.Tensor`): Current state of the environment
            deterministic (bool): Should the policy be deterministic or stochastic

        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        action = self.get_greedy_action(state)
        if not deterministic:
            if random.random() < self.epsilon:
                action = self.env.sample()
        return action

    def _reshape_batch(self, batch: List):
        """Function to reshape experiences for DQN

        Most of the DQN experiences need to be reshaped before sending to the
        Neural Networks
        """
        states = batch[0]
        actions = batch[1].unsqueeze(-1).long()
        rewards = batch[2]
        next_states = batch[3]
        dones = batch[4]

        return states, actions, rewards, next_states, dones

    def get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get Q values corresponding to specific states and actions

        Args:
            states (:obj:`torch.Tensor`): States for which Q-values need to be found
            actions (:obj:`torch.Tensor`): Actions taken at respective states

        Returns:
            q_values (:obj:`torch.Tensor`): Q values for the given states and actions
        """
        q_values = self.model(states)
        q_values = q_values.gather(2, actions)
        return q_values

    def get_target_q_values(
        self, next_states: torch.Tensor, rewards: List[float], dones: List[bool]
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
        # Next Q-values according to target model
        next_q_target_values = self.target_model(next_states)
        # Maximum of next q_target values
        max_next_q_target_values = next_q_target_values.max(2)[0]
        target_q_values = rewards + self.gamma * torch.mul(  # Expected Target Q values
            max_next_q_target_values, (1 - dones)
        )
        # Needs to be unsqueezed to match dimension of q_values
        return target_q_values.unsqueeze(-1)

    def update_params(self, update_interval: int) -> None:
        """Update parameters of the model

        Args:
            update_interval (int): Interval between successive updates of the target model
        """
        self.update_target_model()

        for timestep in range(update_interval):
            batch = self.sample_from_buffer()
            loss = self.get_q_loss(batch)
            self.logs["value_loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # In case the model uses Noisy layers, we must reset the noise every timestep
            if self.noisy:
                self.model.reset_noise()
                self.target_model.reset_noise()

    def calculate_epsilon_by_frame(self) -> float:
        """Helper function to calculate epsilon after every timestep

        Exponentially decays exploration rate from max epsilon to min epsilon
        The greater the value of epsilon_decay, the slower the decrease in epsilon
        """
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(
            -1.0 * self.timestep / self.epsilon_decay
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr": self.lr_value,
            "replay_size": self.replay_size,
            "weights": self.model.state_dict(),
            "timestep": self.timestep,
        }
        return hyperparams, self.model.state_dict()

    def load_weights(self, weights) -> None:
        """Load weights for the agent from pretrained model

        Args:
            weights (:obj:`torch.Tensor`): neural net weights
        """
        self.model.load_state_dict(weights)

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "value_loss": safe_mean(self.logs["value_loss"]),
            "epsilon": safe_mean(self.logs["epsilon"]),
        }
        self.empty_logs()
        return logs

    def empty_logs(self) -> None:
        """Empties logs"""
        self.logs = {}
        self.logs["value_loss"] = []
        self.logs["epsilon"] = []
