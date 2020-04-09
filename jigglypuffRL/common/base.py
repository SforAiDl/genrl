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

    def forward(self, s):
        return self.model(s)

    def get_action(self, s):
        _s = self.forward(s)

        if self.disc:
            _s = nn.Softmax(dim=-1)(_s)
            if self.det:
                a = torch.argmax(_s, dim=-1)
            else:
                c = Categorical(probs=_s)
                a = (c.sample(), c)
        else:
            _s = nn.Tanh()(_s) * self.a_lim
            if self.det:
                a = _s
            else:
                c = Normal(_s, self.a_var)
                a = (c.sample(), c)
        return a


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

    def get_action(self, s):
        s = torch.as_tensor(s).float()
        return self.actor.get_action(s)

    def get_value(self, x):
        x = torch.as_tensor(x).float()
        return self.critic.get_value(x)
