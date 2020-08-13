import os
import sys
from typing import Any, Dict, List

from torch.utils.tensorboard import SummaryWriter


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

    def write(self, kvs: Dict[str, Any]) -> None:
        """
        Add entry to logger

        :param kvs: Entries to be logged
        :type kvs: dict
        """
        for key, value in kvs.items():
            self.writer.add_scalar(key, value, kvs["Episode"])

    def close(self) -> None:
        """
        Close the logger
        """
        self.writer.close()
