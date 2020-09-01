import sys

import numpy as np
from simple_spread_test import make_env
from utils import TensorboardLogger

from contrib.maa2c.a2c_agent import A2CAgent

sys.path.append("add path to /genrl/genrl")  # add path


def train(max_episodes, timesteps_per_eps, logdir):

    tensorboard_logger = TensorboardLogger(logdir)

    environment = make_env(scenario_name="simple_spread", benchmark=False)

    a2c_agent = A2CAgent(environment, 2e-4, 0.99, None, 0.008, timesteps_per_eps)

    for episode in range(1, max_episodes + 1):

        print("EPISODE", episode)

        states = np.asarray(environment.reset())

        values, dones = a2c_agent.collect_rollouts(states)

        a2c_agent.get_traj_loss(values, dones)

        a2c_agent.update_policy()

        params = a2c_agent.get_logging_params()
        params["episode"] = episode

        tensorboard_logger.write(params)

    tensorboard_logger.close()


if __name__ == "__main__":

    train(max_episodes=10000, timesteps_per_eps=300, logdir="./runs/")
