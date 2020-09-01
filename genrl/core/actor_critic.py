from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Categorical, Normal

from genrl.core.base import BaseActorCritic
from genrl.core.policies import MlpPolicy
from genrl.core.values import MlpValue
from genrl.utils.utils import cnn, mlp


class MlpActorCritic(BaseActorCritic):
    """MLP Actor Critic

    Attributes:
        state_dim (int): State dimensions of the environment
        action_dim (int): Action space dimensions of the environment
        hidden (:obj:`list` or :obj:`tuple`): Hidden layers in the MLP
        val_type (str): Value type of the critic network
        discrete (bool): True if the action space is discrete, else False
        sac (bool): True if a SAC-like network is needed, else False
        activation (str): Activation function to be used. Can be either "tanh" or "relu"
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        shared_layers: None,
        policy_layers: Tuple = (32, 32),
        value_layers: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        **kwargs,
    ):
        super(MlpActorCritic, self).__init__()

        self.actor = MlpPolicy(state_dim, action_dim, policy_layers, discrete, **kwargs)
        self.critic = MlpValue(state_dim, action_dim, val_type, value_layers, **kwargs)

    def get_params(self):
        actor_params = self.actor.parameters()
        critic_params = self.critic.parameters()
        return actor_params, critic_params


class MlpSharedActorCritic(BaseActorCritic):
    """MLP Shared Actor Critic

    Attributes:
        state_dim (int): State dimensions of the environment
        action_dim (int): Action space dimensions of the environment
        hidden (:obj:`list` or :obj:`tuple`): Hidden layers in the MLP
        val_type (str): Value type of the critic network
        discrete (bool): True if the action space is discrete, else False
        sac (bool): True if a SAC-like network is needed, else False
        activation (str): Activation function to be used. Can be either "tanh" or "relu"
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        shared_layers: Tuple = (32, 32),
        policy_layers: Tuple = (32, 32),
        value_layers: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        **kwargs,
    ):
        super(MlpSharedActorCritic, self).__init__()
        self.shared = mlp([state_dim] + list(shared_layers))
        self.actor = MlpPolicy(
            shared_layers[-1], action_dim, policy_layers, discrete, **kwargs
        )
        self.critic = MlpValue(
            shared_layers[-1], action_dim, val_type, value_layers, **kwargs
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_params(self):
        actor_params = list(self.shared.parameters()) + list(self.actor.parameters())
        critic_params = list(self.shared.parameters()) + list(self.critic.parameters())
        return actor_params, critic_params

    def get_features(self, state: torch.Tensor):
        """Extract features from the state, which is then an input to get_action and get_value

        Args:
            state (:obj:`torch.Tensor`): The state(s) being passed

        Returns:
            features (:obj:`torch.Tensor`): The feature(s) extracted from the state
        """
        features = self.shared(state)
        return features

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        state = torch.as_tensor(state).float()
        features = self.get_features(state)
        action_probs = self.actor(features)
        action_probs = nn.Softmax(dim=-1)(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1).unsqueeze(-1).float()
            distribution = None
        else:
            distribution = Categorical(probs=action_probs)
            action = distribution.sample()

        return action, distribution

    def get_value(self, state: torch.Tensor):
        state = torch.as_tensor(state).float()
        if self.critic.val_type == "Qsa":
            features = self.shared(state[:, :, :-1])
            features = torch.cat([features, state[:, :, -1].unsqueeze(-1)], dim=-1)
            value = self.critic(features).float().squeeze(-1)
        else:
            features = self.shared(state)
            value = self.critic(features)
        return value


class MlpSingleActorMultiCritic(BaseActorCritic):
    """MLP Actor Critic

    Attributes:
        state_dim (int): State dimensions of the environment
        action_dim (int): Action space dimensions of the environment
        hidden (:obj:`list` or :obj:`tuple`): Hidden layers in the MLP
        val_type (str): Value type of the critic network
        discrete (bool): True if the action space is discrete, else False
        num_critics (int): Number of critics in the architecture
        sac (bool): True if a SAC-like network is needed, else False
        activation (str): Activation function to be used. Can be either "tanh" or "relu"
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        policy_layers: Tuple = (32, 32),
        value_layers: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        num_critics: int = 2,
        **kwargs,
    ):
        super(MlpSingleActorMultiCritic, self).__init__()

        self.num_critics = num_critics

        self.actor = MlpPolicy(state_dim, action_dim, policy_layers, discrete, **kwargs)
        self.critic1 = MlpValue(state_dim, action_dim, "Qsa", value_layers, **kwargs)
        self.critic2 = MlpValue(state_dim, action_dim, "Qsa", value_layers, **kwargs)

        self.action_scale = kwargs["action_scale"] if "action_scale" in kwargs else 1
        self.action_bias = kwargs["action_bias"] if "action_bias" in kwargs else 0

    def get_params(self):
        actor_params = self.actor.parameters()
        critic_params = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )
        return actor_params, critic_params

    def forward(self, x):
        q1_values = self.critic1(x).squeeze(-1)
        q2_values = self.critic2(x).squeeze(-1)
        return (q1_values, q2_values)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        state = torch.as_tensor(state).float()

        if self.actor.sac:
            mean, log_std = self.actor(state)
            std = log_std.exp()
            distribution = Normal(mean, std)

            action_probs = distribution.rsample()
            log_probs = distribution.log_prob(action_probs)
            action_probs = torch.tanh(action_probs)

            action = action_probs * self.action_scale + self.action_bias

            # enforcing action bound (appendix of SAC paper)
            log_probs -= torch.log(
                self.action_scale * (1 - action_probs.pow(2)) + np.finfo(np.float32).eps
            )
            log_probs = log_probs.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

            action = (action.float(), log_probs, mean)
        else:
            action = self.actor.get_action(state, deterministic=deterministic)

        return action

    def get_value(self, state: torch.Tensor, mode="first") -> torch.Tensor:
        """Get Values from the Critic

        Arg:
            state (:obj:`torch.Tensor`): The state(s) being passed to the critics
            mode (str): What values should be returned. Types:
                "both" --> Both values will be returned
                "min" --> The minimum of both values will be returned
                "first" --> The value from the first critic only will be returned

        Returns:
            values (:obj:`list`): List of values as estimated by each individual critic
        """
        state = torch.as_tensor(state).float()

        if mode == "both":
            values = self.forward(state)
        elif mode == "min":
            values = self.forward(state)
            values = torch.min(*values).squeeze(-1)
        elif mode == "first":
            values = self.critic1(state)
        else:
            raise KeyError("Mode doesn't exist")

        return values


class MlpSharedSingleActorMultiCritic(BaseActorCritic):
    """MLP Actor Critic

    Attributes:
        state_dim (int): State dimensions of the environment
        action_dim (int): Action space dimensions of the environment
        hidden (:obj:`list` or :obj:`tuple`): Hidden layers in the MLP
        val_type (str): Value type of the critic network
        discrete (bool): True if the action space is discrete, else False
        num_critics (int): Number of critics in the architecture
        sac (bool): True if a SAC-like network is needed, else False
        activation (str): Activation function to be used. Can be either "tanh" or "relu"
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        shared_layers: Tuple = (32, 32),
        policy_layers: Tuple = (32, 32),
        value_layers: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        num_critics: int = 2,
        **kwargs,
    ):
        super(MlpSharedSingleActorMultiCritic, self).__init__()

        self.num_critics = num_critics
        self.shared = mlp([state_dim] + list(shared_layers))
        self.actor = MlpPolicy(
            shared_layers[-1], action_dim, policy_layers, discrete, **kwargs
        )
        self.critic1 = MlpValue(
            shared_layers[-1], action_dim, "Qsa", value_layers, **kwargs
        )
        self.critic2 = MlpValue(
            shared_layers[-1], action_dim, "Qsa", value_layers, **kwargs
        )

        self.action_scale = kwargs["action_scale"] if "action_scale" in kwargs else 1
        self.action_bias = kwargs["action_bias"] if "action_bias" in kwargs else 0

    def get_params(self):
        actor_params = list(self.actor.parameters()) + list(self.shared.parameters())
        critic_params = (
            list(self.critic1.parameters())
            + list(self.critic2.parameters())
            + list(self.shared.parameters())
        )
        return actor_params, critic_params

    def get_features(self, state: torch.Tensor):
        """Extract features from the state, which is then an input to get_action and get_value

        Args:
            state (:obj:`torch.Tensor`): The state(s) being passed

        Returns:
            features (:obj:`torch.Tensor`): The feature(s) extracted from the state
        """
        features = self.shared(state)
        return features

    def forward(self, x):
        q1_values = self.critic1(x).squeeze(-1)
        q2_values = self.critic2(x).squeeze(-1)
        return (q1_values, q2_values)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        state = torch.as_tensor(state).float()
        state = self.get_features(state)

        if self.actor.sac:
            mean, log_std = self.actor(state)
            std = log_std.exp()
            distribution = Normal(mean, std)

            action_probs = distribution.rsample()
            log_probs = distribution.log_prob(action_probs)
            action_probs = torch.tanh(action_probs)

            action = action_probs * self.action_scale + self.action_bias

            # enforcing action bound (appendix of SAC paper)
            log_probs -= torch.log(
                self.action_scale * (1 - action_probs.pow(2)) + np.finfo(np.float32).eps
            )
            log_probs = log_probs.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

            action = (action.float(), log_probs, mean)
        else:
            action = self.actor.get_action(state, deterministic=deterministic)

        return action

    def get_value(self, state: torch.Tensor, mode="first") -> torch.Tensor:
        """Get Values from the Critic

        Arg:
            state (:obj:`torch.Tensor`): The state(s) being passed to the critics
            mode (str): What values should be returned. Types:
                "both" --> Both values will be returned
                "min" --> The minimum of both values will be returned
                "first" --> The value from the first critic only will be returned

        Returns:
            values (:obj:`list`): List of values as estimated by each individual critic
        """
        state = torch.as_tensor(state).float()
        x = self.get_features(state[:, :, :-1])
        state = torch.cat([x, state[:, :, -1].unsqueeze(-1)], dim=-1)

        if mode == "both":
            values = self.forward(state)
        elif mode == "min":
            values = self.forward(state)
            values = torch.min(*values).squeeze(-1)
        elif mode == "first":
            values = self.critic1(state)
        else:
            raise KeyError("Mode doesn't exist")

        return values


class CNNActorCritic(BaseActorCritic):
    """
        CNN Actor Critic

        :param framestack: Number of previous frames to stack together
        :param action_dim: Action dimensions of the environment
        :param fc_layers: Sizes of hidden layers
        :param val_type: Specifies type of value function: (
    "V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a))
        :param discrete: True if action space is discrete, else False
        :param framestack: Number of previous frames to stack together
        :type action_dim: int
        :type fc_layers: tuple or list
        :type val_type: str
        :type discrete: bool
    """

    def __init__(
        self,
        framestack: int,
        action_dim: spaces.Space,
        policy_layers: Tuple = (256,),
        value_layers: Tuple = (256,),
        val_type: str = "V",
        discrete: bool = True,
        *args,
        **kwargs,
    ):
        super(CNNActorCritic, self).__init__()

        self.feature, output_size = cnn((framestack, 16, 32))
        self.actor = MlpPolicy(
            output_size, action_dim, policy_layers, discrete, **kwargs
        )
        self.critic = MlpValue(output_size, action_dim, val_type, value_layers)

    def get_params(self):
        actor_params = list(self.feature.parameters()) + list(self.actor.parameters())
        critic_params = list(self.feature.parameters()) + list(self.critic.parameters())
        return actor_params, critic_params

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
                Get action from the Actor based on input

                :param state: The state being passed as input to the Actor
                :param deterministic: (True if the action space is deterministic,
        else False)
                :type state: Tensor
                :type deterministic: boolean
                :returns: action
        """
        state = self.feature(state)
        state = state.view(state.size(0), -1)

        action_probs = self.actor(state)
        action_probs = nn.Softmax(dim=-1)(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            distribution = None
        else:
            distribution = Categorical(probs=action_probs)
            action = distribution.sample()

        return action, distribution

    def get_value(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Get value from the Critic based on input

        :param inp: Input to the Critic
        :type inp: Tensor
        :returns: value
        """
        inp = self.feature(inp)
        inp = inp.view(inp.size(0), -1)

        value = self.critic(inp).squeeze(-1)
        return value


actor_critic_registry = {
    "mlp": MlpActorCritic,
    "cnn": CNNActorCritic,
    "mlp12": MlpSingleActorMultiCritic,
    "mlps": MlpSharedActorCritic,
    "mlp12s": MlpSharedSingleActorMultiCritic,
}


def get_actor_critic_from_name(name_: str):
    """
    Returns Actor Critic given the type of the Actor Critic

    :param ac_name: Name of the policy needed
    :type ac_name: str
    :returns: Actor Critic class to be used
    """
    if name_ in actor_critic_registry:
        return actor_critic_registry[name_]
    raise NotImplementedError
