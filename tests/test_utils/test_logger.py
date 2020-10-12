import shutil

from genrl.utils import Logger


class TestLogger:
    def test_loggers(self):
        logger = Logger("./logs", formats=["csv", "stdout", "tensorboard"])
        logger.write({"hello": 0000, "timestep": 10}, log_key="timestep")
        logger.close()
        shutil.rmtree("./logs")
