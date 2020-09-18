import torch
import torch.nn as nn


class VanillaPolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()

    def forward(self, rollout: ):
        return -1*torch.mean(rollout.log_probs)


class PolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()
    
    def forward(self, log_probs: torch.Tensor, phi: torch.Tensor):
        return -1*torch.mean(log_probs*phi)


class DeterministicPolicyGradientLoss(nn.Module):
    def __init__(self):
        super(DeterministicPolicyGradientLoss, self).__init__()

    def forward(self, )


class TDLoss(nn.Module):
    def __init__(self):
        super(TDLoss, self).__init__()

    def forward(self,)