import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .common import TransitionDB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_features))
            self.b_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.w_mu.data.normal_(0, 0.1)
        self.w_sigma.data.normal_(0, 0.1)
        if self.bias:
            self.b_mu.data.normal_(0, 0.1)
            self.b_sigma.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor, frozen: bool = False) -> torch.Tensor:
        if frozen:
            w = self.w_mu
            if bias:
                b = self.b_mu
        else:
            w = self.w_mu + self.w_sigma * torch.randn_like(self.w_sigma)
            if self.bias:
                b = self.b_mu + self.b_sigma * torch.randn_like(self.b_sigma)

        return F.linear(x, w, b)
