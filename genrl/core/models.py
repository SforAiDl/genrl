import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, vae_layers, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, vae_layers[0])
        self.e2 = nn.Linear(vae_layers[0], vae_layers[0])

        self.mean = nn.Linear(vae_layers[0], vae_layers[1])
        self.log_std = nn.Linear(vae_layers[0], vae_layers[1])

        self.d1 = nn.Linear(state_dim + vae_layers[1], vae_layers[0])
        self.d2 = nn.Linear(vae_layers[0], vae_layers[0])
        self.d3 = nn.Linear(vae_layers[0], action_dim)

        self.max_action = max_action
        self.vae_layers = vae_layers[1]
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], dim=-1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((*state.shape[:-1], self.vae_layers))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )

        a = F.relu(self.d1(torch.cat([state, z], dim=-1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
