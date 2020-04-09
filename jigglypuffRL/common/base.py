import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class BasePolicy(nn.Module):
    def __init__(self, discrete, deterministic, **kwargs):
        super(BasePolicy, self).__init__()

        self.discrete = discrete
        self.deterministic = deterministic
        self.action_lim = kwargs["action_lim"] if "action_lim" in kwargs else 1.0
        self.action_var = kwargs["action_var"] if "action_var" in kwargs else 0.1

        self.model = None

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        action_probs = self.forward(state)

        if self.discrete:
            action_probs = nn.Softmax(dim=-1)(action_probs)
            if self.deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                distribution = Categorical(probs=action_probs)
                action = (distribution.sample(), distribution)
        else:
            action_probs = nn.Tanh()(action_probs) * self.action_lim
            if self.deterministic:
                action = action_probs
            else:
                distribution = Normal(action_probs, self.action_var)
                action = (distribution.sample(), distribution)
        return action


class BaseValue(nn.Module):
    def __init__(self):
        super(BaseValue, self).__init__()

        self.model = None

    def forward(self, x):
        return self.model(x)

    def get_value(self, x):
        return self.forward(x).squeeze()


class BaseActorCritic(nn.Module):
    def __init__(self, disc, det):
        super(BaseActorCritic, self).__init__()

        self.actor = None
        self.critic = None

    def get_action(self, state):
        state = torch.as_tensor(state).float()
        return self.actor.get_action(state)

    def get_value(self, state):
        state = torch.as_tensor(state).float()
        return self.critic.get_value(state)
