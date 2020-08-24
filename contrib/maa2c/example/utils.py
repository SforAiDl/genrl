import os
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict

class TensorboardLogger:
    """
    Tensorboard Logging class

    :param logdir: Directory to save log at
    :type logdir: string
    """

    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def write(self, kvs: Dict[str, Any], log_key: str = "episode") -> None:
        """
        Add entry to logger

        :param kvs: Entries to be logged
        :param log_key: Key plotted on x_axis
        :type kvs: dict
        :type log_key: str
        """
        for key, value in kvs.items():
            self.writer.add_scalar(key, value, kvs[log_key])

    def close(self) -> None:
        """
        Close the logger
        """
        self.writer.close()