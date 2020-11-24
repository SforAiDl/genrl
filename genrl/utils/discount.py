from typing import Any

import numpy as np
import torch


def compute_returns_and_advantage(
    rollout_buffer: Any,
    last_value: torch.Tensor,
    dones: np.ndarray,
    use_gae: bool = False,
) -> None:
    """
    Post-processing function: compute the returns (sum of discounted rewards)
    and advantage (A(s) = R - V(S)).
    Adapted from Stable-Baselines PPO2.

    Args:
        rollout_buffer: An instance of the rollout buffer used for OnPolicy Agents
        last_value: (:obj: torch.tensor)
        dones: (:obj: np.ndarray)
        use_gae: (bool) True if Generalized Advantage Estimation is to be used, else False

    Returns:
        A modified Rollout Buffer with advantages calculated
    """
    last_value = last_value.flatten()

    if use_gae:
        gae_lambda = rollout_buffer.gae_lambda
    else:
        gae_lambda = 1.0

    next_value = last_value
    next_non_terminal = 1.0 - dones

    running_advantage = 0.0
    for step in reversed(range(rollout_buffer.buffer_size)):
        delta = (
            rollout_buffer.rewards[step]
            + rollout_buffer.gamma * next_value * next_non_terminal
            - rollout_buffer.values[step]
        )
        running_advantage = (
            delta
            + rollout_buffer.gamma * gae_lambda * next_non_terminal * running_advantage
        )
        next_non_terminal = 1 - rollout_buffer.dones[step]
        next_value = rollout_buffer.values[step]
        rollout_buffer.advantages[step] = running_advantage

    rollout_buffer.returns = rollout_buffer.advantages + rollout_buffer.values

    return rollout_buffer.returns, rollout_buffer.advantages
