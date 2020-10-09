from typing import List, Type, Union

import numpy as np

from genrl.core import PrioritizedBuffer, ReplayBuffer
from genrl.trainers import Trainer
from genrl.utils import safe_mean


class OffPolicyTrainer(Trainer):
    """Off Policy Trainer Class

    Trainer class for all the Off Policy Agents: DQN (all variants), DDPG, TD3 and SAC

    Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        buffer (object): Replay Buffer object
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

    def __init__(
        self,
        *args,
        buffer: Union[Type[ReplayBuffer], Type[PrioritizedBuffer]] = None,
        max_ep_len: int = 500,
        max_timesteps: int = 5000,
        start_update: int = 1000,
        warmup_steps: int = 1000,
        update_interval: int = 50,
        **kwargs
    ):
        super(OffPolicyTrainer, self).__init__(
            *args, off_policy=True, max_timesteps=max_timesteps, **kwargs
        )

        self.max_ep_len = max_ep_len
        self.warmup_steps = warmup_steps
        self.start_update = start_update
        self.update_interval = update_interval
        self.network = self.agent.network

        if buffer is None:
            if self.agent.replay_buffer is None:
                raise Exception("Off Policy Training requires a Replay Buffer")
        else:
            self.agent.replay_buffer = buffer
        self.buffer = self.agent.replay_buffer

    def noise_reset(self) -> None:
        """Resets the agent's action noise functions"""
        if "noise" in self.agent.__dict__ and self.agent.noise is not None:
            self.agent.noise.reset()

    def get_action(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Gets the action to be performed on the environment

        For the first few timesteps (Warmup steps) it selects an action randomly to introduce
        stochasticity to the environment start position.

        Args:
            state (:obj:`np.ndarray`): Current state of the environment
            timestep (int): Current timestep of training

        Returns:
            action (:obj:`np.ndarray`): Action to be taken on the env
        """
        if timestep < self.warmup_steps:
            action = self.env.sample()
        else:
            action = self.agent.select_action(state)
        return action

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
                "Episode Reward": safe_mean(self.training_rewards),
            },
            self.log_key,
        )
        self.training_rewards = []

    def check_game_over_status(self, dones: List[bool]) -> bool:
        """Takes care of game over status of envs

        Whenever an env shows done, the reward accumulated is stored in a list
        and the env is reset. Note that not all envs in the Vectorised Env are reset.

        Args:
            dones (:obj:`list`): Game over statuses of all envs

        Return:
            game_over (bool): True, if at least one environment was done. Else, False
        """
        game_over = False

        for i, done_i in enumerate(dones):
            if done_i:
                self.training_rewards.append(
                    self.env.episode_reward[i].detach().clone()
                )
                self.env.reset_single_env(i)
                self.episodes += 1
                game_over = True

        return game_over

    def train(self) -> None:
        """Main training method"""
        if self.load_weights is not None or self.load_hyperparams is not None:
            self.load()

        state = self.env.reset()
        self.noise_reset()

        self.training_rewards = []
        self.episodes = 0

        for timestep in range(0, self.max_timesteps, self.env.n_envs):
            self.agent.update_params_before_select_action(timestep)

            action = self.get_action(state, timestep)
            next_state, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

            # true_dones contains the "true" value of the dones (game over statuses). It is set
            # to False when the environment is not actually done but instead reaches the max
            # episode length.
            true_dones = [info[i]["done"] for i in range(self.env.n_envs)]
            self.buffer.push((state, action, reward, next_state, true_dones))

            state = next_state.detach().clone()

            if self.check_game_over_status(done):
                self.noise_reset()

                if self.episodes % self.log_interval == 0:
                    self.log(timestep)

                if self.episodes == self.epochs:
                    break

            if timestep >= self.start_update and timestep % self.update_interval == 0:
                self.agent.update_params(self.update_interval)

            if (
                timestep >= self.start_update
                and self.save_interval != 0
                and timestep % self.save_interval == 0
            ):
                self.save(timestep)

        self.env.close()
        self.logger.close()
