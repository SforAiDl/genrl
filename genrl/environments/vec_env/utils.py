from typing import Tuple

import torch


class RunningMeanStd:
    """
    Utility Function to compute a running mean and variance calculator

    :param epsilon: Small number to prevent division by zero for calculations
    :param shape: Shape of the RMS object
    :type epsilon: float
    :type shape: Tuple
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = torch.zeros(shape).double()
        self.var = torch.ones(shape).double()
        self.count = epsilon

    def update(self, batch: torch.Tensor):
        batch_mean = torch.mean(batch, axis=0)
        batch_var = torch.var(batch, axis=0)
        batch_count = batch.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total_count
        M2 = (
            self.var * self.count
            + batch_var * batch_count
            + (delta ** 2) * self.count * batch_count / total_count
        )

        self.mean = new_mean
        self.var = M2 / (total_count - 1)
        self.count = total_count
