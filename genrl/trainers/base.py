import os
from abc import ABC
from typing import Any, List, Optional, Union

import gym
import numpy as np
import torch

from genrl.environments.vec_env import VecEnv
from genrl.utils import Logger, set_seeds


class Trainer(ABC):
    """Base Trainer Class

    To be inherited specific use-cases

    Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        log_mode (:obj:`list` of str): List of different kinds of logging. Supported: ["csv", "stdout", "tensorboard"]
        log_key (str): Key plotted on x_axis. Supported: ["timestep", "episode"]
        log_interval (int): Timesteps between successive logging of parameters onto the console
        logdir (str): Directory where log files should be saved.
        epochs (int): Total number of epochs to train for
        max_timesteps (int): Maximum limit of timesteps to train for
        off_policy (bool): True if the agent is an off policy agent, False if it is on policy
        save_interval (int): Timesteps between successive saves of the agent's important hyperparameters
        save_model (str): Directory where the checkpoints of agent parameters should be saved
        run_num (int): A run number allotted to the save of parameters
        load_model (str): File to load saved parameter checkpoint from
        render (bool): True if environment is to be rendered during training, else False
        evaluate_episodes (int): Number of episodes to evaluate for
        seed (int): Set seed for reproducibility
    """

    def __init__(
        self,
        agent: Any,
        env: Union[gym.Env, VecEnv],
        log_mode: List[str] = ["stdout"],
        log_key: str = "timestep",
        log_interval: int = 10,
        logdir: str = "logs",
        epochs: int = 50,
        max_timesteps: int = None,
        off_policy: bool = False,
        save_interval: int = 0,
        save_model: str = "checkpoints",
        run_num: int = None,
        load_model: str = None,
        render: bool = False,
        evaluate_episodes: int = 50,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.env = env
        self.log_mode = log_mode
        self.log_key = log_key
        self.log_interval = log_interval
        self.logdir = logdir
        self.epochs = epochs
        self.max_timesteps = max_timesteps
        self.off_policy = off_policy
        self.save_interval = save_interval
        self.save_model = save_model
        self.run_num = run_num
        self.load_model = load_model
        self.render = render
        self.evaluate_episodes = evaluate_episodes

        if seed is not None:
            set_seeds(seed, self.env)

        self.logger = Logger(logdir=logdir, formats=[*log_mode])

    def train(self) -> None:
        """Main training method

        To be defined in inherited classes
        """
        raise NotImplementedError

    def evaluate(self, render: bool = False) -> None:
        """Evaluate performance of Agent

        Args:
            render (bool): Option to render the environment during evaluation
        """
        episode, episode_reward = 0, np.zeros(self.env.n_envs)
        episode_rewards = []
        state = self.env.reset()
        while True:
            if self.off_policy:
                action = self.agent.select_action(state, deterministic=True)
            else:
                action, _, _ = self.agent.select_action(state)

            if isinstance(action, torch.Tensor):
                action = action.numpy()

            next_state, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            episode_reward += reward
            state = next_state
            if np.any(done):
                for i, di in enumerate(done):
                    if di:
                        episode += 1
                        episode_rewards.append(episode_reward[i])
                        episode_reward[i] = 0
            if episode == self.evaluate_episodes:
                print(
                    "Evaluated for {} episodes, Mean Reward: {:.2f}, Std Deviation for the Reward: {:.2f}".format(
                        self.evaluate_episodes,
                        np.mean(episode_rewards),
                        np.std(episode_rewards),
                    )
                )
                return

    def save(self, timestep: int) -> None:
        """Function to save all relevant parameters of a given agent

        Args:
            timestep: The timestep during training at which model is being saved
        """
        algo_name = self.agent.__class__.__name__
        env_name = self.env.unwrapped.spec.id

        directory = self.save_model
        path = "{}/{}_{}".format(directory, algo_name, env_name)

        run_num = self.run_num

        if run_num is None:
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
            self.run_num = run_num

        torch.save(
            self.agent.get_hyperparams(),
            "{}/{}-log-{}.pt".format(path, run_num, timestep),
        )

    def load(self):
        """Function to load saved parameters of a given agent"""
        path = self.load_model
        try:
            self.checkpoint = torch.load(path)
        except FileNotFoundError:
            raise Exception("Invalid File Name")

        weights = {}

        for key, item in self.checkpoint.items():
            if "weights" not in key:
                setattr(self, key, item)
            else:
                weights[key] = item

        self.agent.load_weights(weights)

        print("Loaded Pretrained Model!")

    @property
    def n_envs(self) -> int:
        """Number of environments"""
        return self.env.n_envs
