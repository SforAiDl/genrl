from genrl import Logger
import shutil


def test_loggers():
    logger = Logger("./logs", formats=["csv", "stdout", "tensorboard"])
    logger.write({"hello": 0000, "timestep": 10})
    logger.close()
    shutil.rmtree("./logs")
