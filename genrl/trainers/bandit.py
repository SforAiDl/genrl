from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List

import numpy as np

from genrl.utils import Logger


class BanditTrainer(ABC):
    """Bandit Trainer Class

    Args:
        agent (genrl.deep.bandit.dcb_agents.DCBAgent): Agent to train.
        bandit (genrl.deep.bandit.data_bandits.DataBasedBandit): Bandit to train agent on.
        logdir (str): Path to directory to store logs in.
        log_mode (List[str]): List of modes for logging.
    """

    def __init__(
        self,
        agent: Any,
        bandit: Any,
        logdir: str = "./logs",
        log_mode: List[str] = ["stdout"],
    ):
        self.agent = agent
        self.bandit = bandit
        self.logdir = logdir
        self.log_mode = log_mode
        self.logger = Logger(logdir=logdir, formats=[*log_mode])

    @abstractmethod
    def train(self) -> None:
        """
        To be defined in inherited classes
        """


class MABTrainer(BanditTrainer):
    def __init__(
        self,
        agent: Any,
        bandit: Any,
        logdir: str = "./logs",
        log_mode: List[str] = ["stdout"],
    ):
        super(MABTrainer, self).__init__(
            agent, bandit, logdir=logdir, log_mode=log_mode
        )

    def train(self, timesteps, log_every=100) -> None:
        """Train the agent.

        Args:
            timesteps (int, optional): Number of timesteps to train for. Defaults to 10_000.
            log_every (int, optional): Timesteps interval for logging. Defaults to 100.

        Returns:
            dict: Dictionary of metrics recorded during training.
        """

        start_time = datetime.now()
        print(
            f"\nStarted at {start_time:%d-%m-%y %H:%M:%S}\n"
            f"Training {self.agent.__class__.__name__} on {self.bandit.__class__.__name__} "
            f"for {timesteps} timesteps"
        )
        mv_len = timesteps // 20
        context = self.bandit.reset()
        regret_mv_avgs = []
        reward_mv_avgs = []

        for t in range(1, timesteps + 1):
            action = self.agent.select_action(context)
            context, reward = self.bandit.step(action)
            self.agent.action_hist.append(action)
            self.agent.update_params(context, action, reward)
            regret_mv_avgs.append(np.mean(self.bandit.regret_hist[-mv_len:]))
            reward_mv_avgs.append(np.mean(self.bandit.reward_hist[-mv_len:]))
            if t % log_every == 0:
                self.logger.write(
                    {
                        "timestep": t,
                        "regret/regret": self.bandit.regret_hist[-1],
                        "reward/reward": reward,
                        "regret/cumulative_regret": self.bandit.cum_regret,
                        "reward/cumulative_reward": self.bandit.cum_reward,
                        "regret/regret_moving_avg": regret_mv_avgs[-1],
                        "reward/reward_moving_avg": reward_mv_avgs[-1],
                    }
                )

        self.logger.close()
        print(
            f"Training completed in {(datetime.now() - start_time).seconds} seconds\n"
            f"Final Regret Moving Average: {regret_mv_avgs[-1]} | "
            f"Final Reward Moving Average: {reward_mv_avgs[-1]}"
        )

        return {
            "regrets": self.bandit.regret_hist,
            "rewards": self.bandit.reward_hist,
            "cumulative_regrets": self.bandit.cum_regret_hist,
            "cumulative_rewards": self.bandit.cum_reward_hist,
            "regret_moving_avgs": regret_mv_avgs,
            "reward_moving_avgs": reward_mv_avgs,
        }


class DCBTrainer(BanditTrainer):
    def __init__(
        self,
        agent: Any,
        bandit: Any,
        logdir: str = "./logs",
        log_mode: List[str] = ["stdout"],
    ):
        super(DCBTrainer, self).__init__(
            agent, bandit, logdir=logdir, log_mode=log_mode
        )

    def train(self, timesteps: int, **kwargs) -> None:
        """Train the agent.

        Args:
            timesteps (int, optional): Number of timesteps to train for. Defaults to 10_000.
            update_interval (int, optional): Number of timesteps between each successive
                parameter update of the agent. Defaults to 20.
            update_after (int, optional): Number of initial timesteps to start updating
                the agent's parameters after. Defaults to 500.
            batch_size (int, optional): Size of batch to update the agent with. Defaults to 64.
            train_epochs (int, optional): Number of epochs to train agent's model for in
                each update. Defaults to 20.
            log_every (int, optional): Timesteps interval for logging. Defaults to 100.
            ignore_init (int, optional): Number of initial steps to ignore for logging. Defaults to 0.
            init_train_epochs (Optional[int], optional): Initial number of epochs to train agents
                for. Defaults to None which implies `train_epochs` is to be used.
            train_epochs_decay_steps (Optional[int], optional): Steps to decay number of epochs
                to train agent for over. Defaults to None.

        Returns:
            dict: Dictionary of metrics recorded during training.
        """

        update_interval = kwargs.get("update_interval", 20)
        update_after = kwargs.get("update_after", 500)
        train_epochs = kwargs.get("train_epochs", 20)
        log_every = kwargs.get("log_every", 100)
        ignore_init = kwargs.get("ignore_init", 0)
        init_train_epochs = kwargs.get("init_train_epochs", None)
        train_epochs_decay_steps = kwargs.get("train_epochs_decay_steps", None)

        start_time = datetime.now()
        print(
            f"\nStarted at {start_time:%d-%m-%y %H:%M:%S}\n"
            f"Training {self.agent.__class__.__name__} on {self.bandit.__class__.__name__} "
            f"for {timesteps} timesteps"
        )
        mv_len = timesteps // 20
        context = self.bandit.reset()
        regret_mv_avgs = []
        reward_mv_avgs = []

        train_epochs_schedule = None
        if init_train_epochs is not None and train_epochs_decay_steps is not None:
            train_epochs_schedule = np.linspace(
                init_train_epochs, train_epochs, train_epochs_decay_steps
            )

        for t in range(1, timesteps + 1):
            action = self.agent.select_action(context)
            new_context, reward = self.bandit.step(action)
            self.agent.update_db(context, action, reward)
            context = new_context

            if train_epochs_schedule is not None and t < train_epochs_decay_steps:
                train_epochs = int(train_epochs_schedule[t])

            if t > update_after and t % update_interval == 0:
                self.agent.update_params(
                    action, kwargs.get("batch_size", 64), train_epochs
                )

            if t > ignore_init:
                regret_mv_avgs.append(np.mean(self.bandit.regret_hist[-mv_len:]))
                reward_mv_avgs.append(np.mean(self.bandit.reward_hist[-mv_len:]))
                if t % log_every == 0:
                    self.logger.write(
                        {
                            "timestep": t,
                            "regret/regret": self.bandit.regret_hist[-1],
                            "reward/reward": reward,
                            "regret/cumulative_regret": self.bandit.cum_regret,
                            "reward/cumulative_reward": self.bandit.cum_reward,
                            "regret/regret_moving_avg": regret_mv_avgs[-1],
                            "reward/reward_moving_avg": reward_mv_avgs[-1],
                        }
                    )

        self.logger.close()
        print(
            f"Training completed in {(datetime.now() - start_time).seconds} seconds\n"
            f"Final Regret Moving Average: {regret_mv_avgs[-1]} | "
            f"Final Reward Moving Average: {reward_mv_avgs[-1]}"
        )

        return {
            "regrets": self.bandit.regret_hist,
            "rewards": self.bandit.reward_hist,
            "cumulative_regrets": self.bandit.cum_regret_hist,
            "cumulative_rewards": self.bandit.cum_reward_hist,
            "regret_moving_avgs": regret_mv_avgs,
            "reward_moving_avgs": reward_mv_avgs,
        }
