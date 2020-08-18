import shutil

from genrl.deep.common.logger import Logger


def test_loggers():
    logger = Logger("./logs", formats=["csv", "stdout", "tensorboard"])
    logger.write({"hello": 0000, "timestep": 10}, log_key="timestep")
    logger.close()
    shutil.rmtree("./logs")
