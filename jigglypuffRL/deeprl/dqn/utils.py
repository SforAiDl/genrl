import torch.nn as nn


class DuelingDQNValueMlp(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNValueMlp, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
