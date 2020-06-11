import numpy as np
from typing import Tuple


class RunningMeanStd:
    """
    Utility Function to compute a running mean and variance calculator

    :param epsilon: Small number to prevent division by zero for calculations
    :param shape: Shape of the RMS object
    :type epsilon: float
    :type shape: Tuple
    """
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray):
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total_count
        M2 = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / total_count
        )

        self.mean = new_mean
        self.var = M2 / (total_count - 1)
        self.count = total_count
