import os
import sys
from typing import Any, Dict, List

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger class to log important information

    :param logdir: Directory to save log at
    :param formats: Formatting of each log ['csv', 'stdout', 'tensorboard']
    :type logdir: string
    :type formats: list
    """

    def __init__(self, logdir: str = None, formats: List[str] = ["csv"]):
        if logdir is None:
            self._logdir = os.getcwd()
        else:
            self._logdir = logdir
            if not os.path.isdir(self._logdir):
                os.makedirs(self._logdir)
        self._formats = formats
        self.writers = []
        for ft in self.formats:
            self.writers.append(get_logger_by_name(ft)(self.logdir))

    def write(self, kvs: Dict[str, Any], log_key: str = "timestep") -> None:
        """
        Add entry to logger

        :param kvs: Entry to be logged
        :param log_key: Key plotted on log_key
        :type kvs: dict
        :type log_key: str
        """
        for writer in self.writers:
            writer.write(kvs, log_key)

    def close(self) -> None:
        """
        Close the logger
        """
        for writer in self.writers:
            writer.close()

    @property
    def logdir(self) -> str:
        """
        Return log directory
        """
        return self._logdir

    @property
    def formats(self) -> List[str]:
        """
        Return save format(s)
        """
        return self._formats


class HumanOutputFormat:
    """
    Output from a log file in a human readable format

    :param logdir: Directory at which log is present
    :type logdir: string
    """

    def __init__(self, logdir: str):
        self.file = os.path.join(logdir, "train.log")
        self.first = True
        self.lens = []
        self.maxlen = 0

    def write(self, kvs: Dict[str, Any], log_key) -> None:
        """
        Log the entry out in human readable format

        :param kvs: Entries to be logged
        :type kvs: dict
        """
        self.write_to_file(kvs, sys.stdout)
        with open(self.file, "a") as file:
            self.write_to_file(kvs, file)

    def write_to_file(self, kvs: Dict[str, Any], file=sys.stdout) -> None:
        """
        Log the entry out in human readable format

        :param kvs: Entries to be logged
        :param file: Name of file to write logs to
        :type kvs: dict
        :type file: io.TextIOWrapper
        """
        if self.first:
            self.first = False
            self.max_key_len(kvs)
            for key, value in kvs.items():
                print(
                    "{}{}".format(str(key), " " * (self.maxlen - len(str(key)))),
                    end="  ",
                    file=file,
                )
            print()
        for key, value in kvs.items():
            rounded = self.round(value)
            print(
                "{}{}".format(rounded, " " * (self.maxlen - len(str(rounded)))),
                end="  ",
                file=file,
            )
        print("", file=file)

    def max_key_len(self, kvs: Dict[str, Any]) -> None:
        """
        Finds max key length

        :param kvs: Entries to be logged
        :type kvs: dict
        """
        self.lens = [len(str(key)) for key, value in kvs.items()]
        maxlen = max(self.lens)
        self.maxlen = maxlen
        if maxlen < 15:
            self.maxlen = 15

    def round(self, num: float) -> float:
        """
        Returns a rounded float value depending on self.maxlen

        :param num: Value to round
        :type num: float
        """
        exponent_len = len(str(num // 1.0)[:-2])
        rounding_len = min(self.maxlen - exponent_len, 4)
        return round(num, rounding_len)

    def close(self) -> None:
        pass


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

    def write(self, kvs: Dict[str, Any], log_key: str = "timestep") -> None:
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


class CSVLogger:
    """
    CSV Logging class

    :param logdir: Directory to save log at
    :type logdir: string
    """

    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        self.file = open("{}/train.csv".format(logdir), "w")
        self.first = True
        self.keynames = {}

    def write(self, kvs: Dict[str, Any], log_key) -> None:
        """
        Add entry to logger

        :param kvs: Entries to be logged
        :type kvs: dict
        """
        if self.first:
            for i, key in enumerate(kvs.keys()):
                self.keynames[key] = i
                self.file.write(key)
                self.file.write(",")
            self.file.write("\n")
            self.first = False

        for i, (key, value) in enumerate(kvs.items()):
            if key not in self.keynames.keys():
                raise Exception(
                    "A new value '{}' cannot be added to CSVLogger".format(key)
                )
            if i != self.keynames[key]:
                raise Exception("Value not at the same index as when initialized")
            self.file.write(str(value))
            self.file.write(",")

        self.file.write("\n")

    def close(self) -> None:
        """
        Close the logger
        """
        self.file.close()


logger_registry = {
    "stdout": HumanOutputFormat,
    "tensorboard": TensorboardLogger,
    "csv": CSVLogger,
}


def get_logger_by_name(name: str):
    """
    Gets the logger given the type of logger

    :param name: Name of the value function needed
    :type name: string
    :returns: Logger
    """
    if name not in logger_registry.keys():
        raise NotImplementedError
    else:
        return logger_registry[name]
