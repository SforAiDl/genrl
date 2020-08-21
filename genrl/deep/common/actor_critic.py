from typing import Tuple

import torch
import torch.nn as nn
from gym import spaces
from torch.distributions import Categorical

from genrl.deep.common.base import BaseActorCritic
from genrl.deep.common.policies import MlpPolicy
from genrl.deep.common.utils import cnn
from genrl.deep.common.values import MlpValue


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
        policy_layers: Tuple = (32, 32),
        value_layers: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        **kwargs,
    ):
        super(MlpActorCritic, self).__init__()

        self.actor = MlpPolicy(state_dim, action_dim, policy_layers, discrete, **kwargs)
        self.critic = MlpValue(state_dim, action_dim, val_type, value_layers, **kwargs)


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
        hidden: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool = True,
        num_critics: int = 2,
        **kwargs,
    ):
        super(MlpSingleActorMultiCritic, self).__init__()

        self.num_critics = num_critics

        self.actor = MlpPolicy(state_dim, action_dim, hidden, discrete, **kwargs)
        self.critic = [
            MlpValue(state_dim, action_dim, val_type, hidden, **kwargs)
            for _ in range(num_critics)
        ]

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
            values = [self.critic[i].get_value(state) for i in range(self.num_critics)]
        elif mode == "min":
            values = [self.critic[i].get_value(state) for i in range(self.num_critics)]
            values = torch.min(values[0], values[1])
        elif mode == "first":
            values = self.critic[0].get_value(state)
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
