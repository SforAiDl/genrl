import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class BasePolicy(nn.Module):
    def __init__(self, disc, det, **kwargs):
        super(BasePolicy, self).__init__()

        self.disc = disc
        self.det = det
        self.a_lim = kwargs["a_lim"] if "a_lim" in kwargs else 1.0
        self.a_var = kwargs["a_var"] if "a_var" in kwargs else 0.1

        self.model = None

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        ps = self.forward(state)

        if self.disc:
            ps = nn.Softmax(dim=-1)(ps)
            if self.det:
                action = torch.argmax(ps, dim=-1)
            else:
                action = Categorical(probs=ps).sample()
        else:
            ps = nn.Tanh()(ps) * self.a_lim
            if self.det:
                action = ps
            else:
                action = Normal(ps, self.a_var).sample()
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
