import torch

def td_target(reward_t: torch.Tensor, value_t: torch.Tensor, gamma: float) -> torch.Tensor:
    #TODO: Check Shapes 

    # Target = R_t + \gamma * V_t

    target = reward_t + gamma * value_t
    return target

def td_error(value_t0: torch.Tensor, reward_t1: torch.Tensor, gamma: float, value_t1: torch.Tensor) -> torch.Tensor:
    #TODO: Check Shapes

    # Delta = V_{t-1} - (R_t + \gamma * V_t)

    target = td_target(reward_t1, value_t1, gamma)
    return value_t0 - target

def value_loss(value_t0: torch.Tensor, reward_t1: torch.Tensor, gamma: float, value_t1: torch.Tensor) -> torch.Tensor:
    #TODO: Check Shapes

    # Loss = 0.5 * (V_{t-1} - (R_t + \gamma * V_t))^2

    delta = td_error(value_t0, reward_t1, gamma, value_t0)
    td_loss = 0.5 * torch.square(delta)
    return td_loss

"""
1. Losses over trajectories? TD Lambda?
2. Consideration of classes than such functions?
"""