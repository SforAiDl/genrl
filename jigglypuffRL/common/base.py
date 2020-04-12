import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class BasePolicy(nn.Module):
    def __init__(self, discrete, state_dim, action_dim, hidden, **kwargs):
        super(BasePolicy, self).__init__()

        self.discrete = discrete
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.action_lim = kwargs["action_lim"] if "action_lim" in kwargs else 1.0
        self.action_var = kwargs["action_var"] if "action_var" in kwargs else 0.1
        self.sac = kwargs["sac"] if "sac" in kwargs else False

        if self.sac:
            self.fc_mean = nn.Linear(self.hidden[-1], self.action_dim)
            self.fc_std = nn.Linear(self.hidden[-1], self.action_dim)

        self.model = None

    def forward(self, state):
        state = self.model.forward(state)
        state = nn.ReLU()(state)
        if self.sac:
            mean = self.fc_mean(state)
            log_std = self.fc_std(state)
            log_std = torch.clamp(log_std, min=-20.0, max=2.0)
            return mean, log_std

        return state

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
        return self.forward(x).squeeze(-1)


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
