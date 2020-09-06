from typing import Optional, Tuple

import torch  # noqa
import torch.nn as nn  # noqa
from torch.distributions import Categorical, Normal


class BasePolicy(nn.Module):
    """
    Basic implementation of a general Policy

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type discrete: bool
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden: Tuple, discrete: bool, **kwargs
    ):
        super(BasePolicy, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.discrete = discrete

        self.action_lim = kwargs["action_lim"] if "action_lim" in kwargs else 1.0
        self.action_var = kwargs["action_var"] if "action_var" in kwargs else 0.1
        self.sac = kwargs["sac"] if "sac" in kwargs else False

        if self.sac:
            self.fc_mean = nn.Linear(self.hidden[-1], self.action_dim)
            self.fc_std = nn.Linear(self.hidden[-1], self.action_dim)

        self.model = None

    def forward(
        self, state: torch.Tensor
    ) -> (Tuple[torch.Tensor, Optional[torch.Tensor]]):
        """
        Defines the computation performed at every call.

        :param state: The state being passed as input to the policy
        :type state: Tensor
        """
        state = self.model.forward(state)
        if self.sac:
            state = nn.ReLU()(state)
            mean = self.fc_mean(state)
            log_std = self.fc_std(state)
            log_std = torch.clamp(log_std, min=-20.0, max=2.0)
            return mean, log_std

        return state

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
                Get action from policy based on input

                :param state: The state being passed as input to the policy
                :param deterministic: (True if the action space is deterministic,
        else False)
                :type state: Tensor
                :type deterministic: boolean
                :returns: action
        """
        action_probs = self.forward(state)

        if self.discrete:
            action_probs = nn.Softmax(dim=-1)(action_probs)
            if deterministic:
                action = (torch.argmax(action_probs, dim=-1), None)
            else:
                distribution = Categorical(probs=action_probs)
                action = (distribution.sample(), distribution)
        else:
            action_probs = nn.Tanh()(action_probs) * self.action_lim
            if deterministic:
                action = (action_probs, None)
            else:
                distribution = Normal(action_probs, self.action_var)
                action = (distribution.sample(), distribution)
        return action


class BaseValue(nn.Module):
    """
    Basic implementation of a general Value function
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(BaseValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        :param state: Input to value function
        :type state: Tensor
        """
        return self.model.forward(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value from value function based on input

        :param state: Input to value function
        :type state: Tensor
        :returns: Value
        """
        return self.forward(state).squeeze(-1)


class BaseActorCritic(nn.Module):
    """
    Basic implementation of a general Actor Critic
    """

    def __init__(self):
        super(BaseActorCritic, self).__init__()

        self.actor = None
        self.critic = None

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
        state = torch.as_tensor(state).float()
        return self.actor.get_action(state, deterministic=deterministic)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value from the Critic based on input

        :param state: Input to the Critic
        :type state: Tensor
        :returns: value
        """
        state = torch.as_tensor(state).float()
        return self.critic.get_value(state)
