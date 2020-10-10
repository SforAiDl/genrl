from typing import List

import numpy as np
import torch

from genrl.trainers.offpolicy import OffPolicyTrainer


class OfflineTrainer(OffPolicyTrainer):
    """Offline RL Trainer Class

    Trainer class for all the Offline RL Agents: BCQ (more to be added)

    Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        buffer (object): Replay Buffer object
        buffer_path (str): Path to the saved buffer file
        max_ep_len (int): Maximum Episode length for training
        max_timesteps (int): Maximum limit of timesteps to train for
        warmup_steps (int): Number of warmup steps. (random actions are taken to add randomness to training)
        start_update (int): Timesteps after which the agent networks should start updating
        update_interval (int): Timesteps between target network updates
        log_mode (:obj:`list` of str): List of different kinds of logging. Supported: ["csv", "stdout", "tensorboard"]
        log_key (str): Key plotted on x_axis. Supported: ["timestep", "episode"]
        log_interval (int): Timesteps between successive logging of parameters onto the console
        logdir (str): Directory where log files should be saved.
        epochs (int): Total number of epochs to train for
        off_policy (bool): True if the agent is an off policy agent, False if it is on policy
        save_interval (int): Timesteps between successive saves of the agent's important hyperparameters
        save_model (str): Directory where the checkpoints of agent parameters should be saved
        run_num (int): A run number allotted to the save of parameters
        load_model (str): File to load saved parameter checkpoint from
        render (bool): True if environment is to be rendered during training, else False
        evaluate_episodes (int): Number of episodes to evaluate for
        seed (int): Set seed for reproducibility
    """

    def __init__(self, *args, buffer_path: str = None, **kwargs):
        super(OfflineTrainer, self).__init__(
            *args, start_update=0, warmup_steps=0, update_interval=1, **kwargs
        )
        self.buffer_path = buffer_path

        if self.buffer_path is None:
            self.generate_buffer("random")

    def generate_buffer(self, generate_type: str = "random") -> None:
        """Make a replay buffer from a specific kind of agent

        Args:
            generate_type (str): Type of generation for the buffer. Can choose from ["random", "agent"]
                Not generatable at the moment.
        """
        raise NotImplementedError

    def check_game_over_status(self, timestep: int) -> bool:
        """Takes care of game over status of envs

        Whenever a trajectory shows done, the reward accumulated is stored in a list

        Args:
            timestep (int): Timestep for which game over condition needs to be checked

        Return:
            game_over (bool): True, if at least one environment was done. Else, False
        """
        game_over = False

        for i, batch_done in enumerate(self.agent.batch.dones):
            for j, done in enumerate(batch_done):
                if done or timestep == self.max_ep_len:
                    self.episodes += 1
                    game_over = True

        return game_over

    def log(self, timestep: int) -> None:
        """Helper function to log

        Sends useful parameters to the logger.

        Args:
            timestep (int): Current timestep of training
        """
        self.logger.write(
            {
                "timestep": timestep,
                "Episode": self.episodes,
                **self.agent.get_logging_params(),
            },
            self.log_key,
        )

    def train(self) -> None:
        """Main training method"""
        self.buffer.load(self.buffer_path)
        self.noise_reset()

        self.training_rewards = []
        self.episodes = 0

        for timestep in range(0, self.max_timesteps):
            self.agent.update_params()

            if timestep % self.log_interval == 0:
                self.log(timestep)

            if self.episodes >= self.epochs:
                break

            if self.save_interval != 0 and timestep % self.save_interval == 0:
                self.save(timestep)

        self.env.close()
        self.logger.close()
