import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CentralizedActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(CentralizedActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.shared_layer_1 = nn.Linear(self.obs_dim, 512)
        torch.nn.init.xavier_uniform_(self.shared_layer_1.weight)

        self.shared_layer_2 = nn.Linear(512, 256)
        torch.nn.init.xavier_uniform_(self.shared_layer_2.weight)

        self.value = nn.Linear(256, 1)
        torch.nn.init.xavier_uniform_(self.value.weight)

        self.policy = nn.Linear(256, self.action_dim)
        torch.nn.init.xavier_uniform_(self.policy.weight)

    def forward(self, x):
        x_s = F.relu(self.shared_layer_1(x))
        x_s = F.relu(self.shared_layer_2(x_s))
        qval = self.value(x_s)
        policy = self.policy(x_s)

        return policy, qval

    def get_action(self, state, device, one_hot=False):
        state = torch.FloatTensor(state).to(device)
        logits, _ = self.forward(state)
        if one_hot:
            logits = self.onehot_from_logits(logits)
            return logits

        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)
        index = probs.sample().cpu().detach().item()
        return index

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ]
        # chooses between best and random actions using epsilon greedy
        return torch.stack(
            [
                argmax_acs[i] if r > eps else rand_acs[i]
                for i, r in enumerate(torch.rand(logits.shape[0]))
            ]
        )
