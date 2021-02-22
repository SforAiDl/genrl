from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from genrl.utils.utils import mlp


class VAE(nn.Module):
    """VAE model to be used

    Currently only used in BCQ

    Attributes:
        state_dim (int): State dimensions of the environment
        action_dim (int): Action space dimensions of the environment
        action_lim (float): Maximum action that can be taken. Used to scale the decoder output to action space
        hidden_layers (:obj:`list` or :obj:`tuple`): Hidden layers in the Encoder and Decoder
            (will be reversed to use in the decoder)
        latent_dim (int): Dimensions of the latent space for the VAE
        activation (str): Activation function to be used. Can be either "tanh" or "relu"
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_lim: float,
        hidden_layers: Tuple = (32, 32),
        latent_dim: int = None,
        activation: str = "relu",
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim if latent_dim is not None else action_dim * 2
        self.max_action = action_lim

        self.encoder = mlp(
            [state_dim + action_dim, hidden_layers], activation=activation
        )

        self.mean = nn.Linear(hidden_layers[-1], self.latent_dim)
        self.log_std = nn.Linear(hidden_layers[-1], self.latent_dim)

        self.decoder = mlp(
            [state_dim + self.latent_dim, hidden_layers[::-1], action_dim],
            activation=activation,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE

        Passes a concatenated vector of the states and actions to an encoder. The latent space is then
        modelled by the variable z. The latent space vector is then passed through the decoder to give
        the VAE-suggested actions.

        Args:
            state (:obj:`torch.Tensor`): State being observed by agent
            action (:obj:`torch.Tensor`): Action being observed by the agent for the respective state

        Returns:
            u (:obj:`torch.Tensor`): VAE-suggested action
            mean (:obj:`torch.Tensor`): Mean of VAE latent space
            std (:obj:`torch.Tensor`): Standard deviation of VAE latent space
        """
        e = F.relu(self.encoder(torch.cat([state, action], dim=-1)))

        mean = self.mean(e)
        log_std = self.log_std(e).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """Decoder output

        Decodes a given state to give an action

        Args:
            state (:obj:`torch.Tensor`): State being observed by agent
            z (:obj:`torch.Tensor`): Latent space vector

        Returns:
            u (:obj:`torch.Tensor`): VAE-suggested action
        """
        if z is None:
            z = (
                torch.randn((*state.shape[:-1], self.latent_dim))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )

        d = F.tanh(self.decoder(torch.cat([state, z], dim=-1)))
        return self.max_action * d
