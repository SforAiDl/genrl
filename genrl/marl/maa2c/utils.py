import os
import random
from typing import Any, List, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn


def save_params(algo: Any, timestep: int) -> None:

    algo_name = algo.__class__.__name__
    if isinstance(algo.env, VecEnv):
        env_name = algo.env.envs[0].unwrapped.spec.id
    else:
        env_name = algo.env.unwrapped.spec.id
    directory = algo.save_model
    path = "{}/{}_{}".format(directory, algo_name, env_name)

    if algo.run_num is not None:
        run_num = algo.run_num
    else:
        if not os.path.exists(path):
            os.makedirs(path)
            run_num = 0
        elif list(os.scandir(path)) == []:
            run_num = 0
        else:
            last_path = sorted(os.scandir(path), key=lambda d: d.stat().st_mtime)[
                -1
            ].path
            run_num = int(last_path[len(path) + 1 :].split("-")[0]) + 1
        algo.run_num = run_num

    torch.save(
        algo.get_hyperparams(), "{}/{}-log-{}.pt".format(path, run_num, timestep)
    )
