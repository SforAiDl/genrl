from abc import ABC
from torchvision import transforms
import torch
import numpy as np
from collections import deque
import gym

from .utils import set_seeds, save_params
from .logger import Logger
from .VecEnv import venv
from .buffers import ReplayBuffer, PrioritizedBuffer
from typing import Union, Type, List, Optional, Any


class Trainer(ABC):
    """
    Base Trainer class. To be inherited specific usecases.

    :param agent: Algorithm object
    :param logger: Logger object
    :param buffer: Buffer Object
    :param off_policy: Is the algorithm off-policy?
    :param save_interval: Model to save in each of these many timesteps
    :param render: Should the Environment render
    :param max_ep_len: Max Episode Length
    :param distributed: True if distributed training is enabled, else \
False (To be implemented)
    :param ckpt_log_name: Model checkpoint name
    :param steps_per_epochs: Steps to take per epoch?
    :param epochs: Total Epochs to train for
    :param device: Device to train model on
    :param log_interval: Log important params every these many steps
    :param batch_size: Size of batch
    :param seed: Set seed for reproducibility
    :param deterministic_actions: Take deterministic actions during training.
    :type agent: object
    :type logger: object
    :type buffer: object
    :type off_policy: bool
    :type save_interval: int
    :type render: bool
    :type max_ep_len: int
    :type distributed: bool
    :type ckpt_log_name: string
    :type steps_per_epochs: int
    :type epochs: int
    :type device: string
    :type log_interval: int
    :type batch_size: int
    :type seed: int
    :type deterministic_actions: bool
    """

    def __init__(
        self,
        agent: Any,
        log_mode: List[str] = ["stdout"],
        buffer: Union[Type[ReplayBuffer], Type[PrioritizedBuffer]] = None,
        off_policy: bool = False,
        save_interval: int = 0,
        render: bool = False,
        max_ep_len: int = 1000,
        distributed: bool = False,
        ckpt_log_name: str = "experiment",
        steps_per_epoch: int = 4000,
        epochs: int = 10,
        device: Union[torch.device, str] = "cpu",
        log_interval: int = 10,
        evaluate_episodes: int = 500,
        logdir: str = "logs",
        batch_size: int = 50,
        seed: Optional[int] = None,
        deterministic_actions: bool = False,
        transform: bool = None,
        history_length: int = 4,
    ):
        self.agent = agent
        self.env = agent.env
        self.log_mode = log_mode
        self.logdir = logdir
        self.off_policy = off_policy
        if self.off_policy and buffer is None:
            if self.agent.replay_buffer is None:
                raise Exception("Off Policy Training requires a Replay Buffer")
            else:
                self.buffer = self.agent.replay_buffer
        self.save_interval = save_interval
        self.render = render
        self.max_ep_len = max_ep_len
        self.ckpt_log_name = ckpt_log_name
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.evaluate_episodes = evaluate_episodes
        self.batch_size = batch_size
        self.deterministic_actions = deterministic_actions
        self.transform = transform
        self.history_length = history_length

        if seed is not None:
            set_seeds(seed, self.env)

        self.logger = Logger(logdir=logdir, formats=[*log_mode])

    def train(self) -> None:
        """
        To be defined in inherited classes
        """
        raise NotImplementedError

    def evaluate(self) -> None:
        """
        Evaluate function
        """
        ep, ep_r = 0, 0
        ep_rews = []
        state = self.env.reset()
        while True:
            if self.agent.__class__.__name__ == "DQN":
                action = self.agent.select_action(state, explore=False)
            else:
                action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            ep_r += reward
            state = next_state
            if done:
                ep += 1
                ep_rews.append(ep_r)
                state = self.env.reset()
                ep_r = 0
                if ep == self.evaluate_episodes:
                    print(
                        "Evaluated for {} episodes, Mean Reward: {}, Std Deviation for the Reward: {}".format(
                            self.evaluate_episodes,
                            np.around(np.mean(ep_rews), decimals=4),
                            np.around(np.std(ep_rews), decimals=4),
                        )
                    )
                    break

    @property
    def n_envs(self) -> int:
        """
        Number of environments
        """
        return self.env.n_envs


class OffPolicyTrainer(Trainer):
    """
    Off-Policy Trainer class

    :param agent: Algorithm object
    :param logger: Logger object
    :param buffer: Buffer Object. Cannot be None for Off-policy
    :param off_policy: Is the algorithm off-policy?
    :param save_interval: Model to save in each of these many timesteps
    :param render: Should the Environment render
    :param max_ep_len: Max Episode Length
    :param distributed: Should distributed training be enabled? \
(To be implemented)
    :param ckpt_log_name: Model checkpoint name
    :param steps_per_epochs: Steps to take per epoch?
    :param epochs: Total Epochs to train for
    :param device: Device to train model on
    :param log_interval: Log important params every these many steps
    :param batch_size: Size of batch
    :param seed: Set seed for reproducibility
    :param deterministic_actions: Take deterministic actions during training.
    :param warmup_steps: Observe the environment for these many steps \
with randomly sampled actions to store in buffer.
    :param start_update: Starting updating the policy after these \
many steps
    :param update_interval: Update model policies after number of steps.
    :type agent: object
    :type logger: object
    :type buffer: object
    :type off_policy: bool
    :type save_interval:int
    :type render: bool
    :type max_ep_len: int
    :type distributed: int
    :type ckpt_log_name: string
    :type steps_per_epochs: int
    :type epochs: int
    :type device: string
    :type log_interval: int
    :type batch_size: int
    :type seed: int
    :type deterministic_actions: bool
    :type warmup_steps: int
    :type start_update: int
    :type update_interval: int
    """

    def __init__(
        self,
        agent: Any,
        log_mode: List[str] = ["stdout"],
        buffer: Union[Type[ReplayBuffer], Type[PrioritizedBuffer]] = None,
        off_policy: bool = True,
        save_interval: int = 0,
        render: bool = False,
        max_ep_len: int = 1000,
        distributed: bool = False,
        ckpt_log_name: str = "experiment",
        steps_per_epoch: int = 4000,
        epochs: int = 10,
        device: Union[torch.device, str] = "cpu",
        log_interval: int = 10,
        evaluate_episodes: int = 500,
        logdir: str = "logs",
        batch_size: int = 50,
        seed: Optional[int] = 0,
        deterministic_actions: bool = False,
        warmup_steps: int = 10000,
        start_update: int = 1000,
        update_interval: int = 50,
    ):
        super(OffPolicyTrainer, self).__init__(
            agent,
            log_mode,
            buffer,
            off_policy,
            save_interval,
            render,
            max_ep_len,
            distributed,
            ckpt_log_name,
            steps_per_epoch,
            epochs,
            device,
            log_interval,
            evaluate_episodes,
            logdir,
            batch_size,
            seed,
            deterministic_actions,
        )
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.start_update = start_update
        self.network_type = self.agent.network_type

    def train(self) -> None:
        """
        Run training
        """
        state, episode_reward, episode_len, episode = self.env.reset(), 0, 0, 0
        total_steps = self.steps_per_epoch * self.epochs
        # self.agent.learn()

        if "noise" in self.agent.__dict__ and self.agent.noise is not None:
            self.agent.noise.reset()

        if self.agent.__class__.__name__ == "DQN":
            self.agent.update_target_model()

        for t in range(total_steps):
            if self.agent.__class__.__name__ == "DQN":
                self.agent.epsilon = self.agent.calculate_epsilon_by_frame(t)

                action = self.agent.select_action(state)

            else:
                if t < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    if self.deterministic_actions:
                        action = self.agent.select_action(state, deterministic=True)
                    else:
                        action = self.agent.select_action(state)

            next_state, reward, done, info = self.env.step(action)
            if self.render:
                self.env.render()

            episode_reward += reward
            episode_len += 1

            done = False if episode_len == self.max_ep_len else done

            self.buffer.push((state, action, reward, next_state, done))
            state = next_state

            if done or (episode_len == self.max_ep_len):
                if "noise" in self.agent.__dict__ and self.agent.noise is not None:
                    self.agent.noise.reset()

                if episode % self.log_interval == 0:
                    self.logger.write(
                        {
                            "timestep": t,
                            "Episode": episode,
                            "Episode Reward": np.around(episode_reward, decimals=4),
                        }
                    )

                state, episode_reward, episode_len = self.env.reset(), 0, 0
                episode += 1

            # update params for DQN
            if self.agent.__class__.__name__ == "DQN":
                if self.agent.replay_buffer.get_len() > self.agent.batch_size:
                    self.agent.update_params()

                if t % self.update_interval == 0:
                    self.agent.update_target_model()

            # update params for other agents
            else:
                if t >= self.start_update and t % self.update_interval == 0:
                    for _ in range(self.update_interval):
                        batch = self.buffer.sample(self.batch_size)
                        states, actions, next_states, rewards, dones = (
                            x.to(self.device) for x in batch
                        )
                        if self.agent.__class__.__name__ == "TD3":
                            self.agent.update_params(
                                states, actions, next_states, rewards, dones, _
                            )
                        else:
                            self.agent.update_params(
                                states, actions, next_states, rewards, dones
                            )

            if (
                t >= self.start_update
                and self.save_interval != 0
                and t % self.save_interval == 0
            ):
                self.checkpoint = self.agent.get_hyperparams()
                save_params(self.agent, t)

        self.env.close()
        self.logger.close()


class OnPolicyTrainer(Trainer):
    """
    Base Trainer class. To be inherited specific usecases.

    :param agent: Algorithm object
    :param logger: Logger Object
    :param buffer: Buffer Object
    :param off_policy: Is the algorithm off-policy?
    :param save_interval: Model to save in each of these many timesteps
    :param render: Should the Environment render
    :param max_ep_len: Max Episode Length
    :param distributed: Should distributed training be enabled? \
(To be implemented)
    :param ckpt_log_name: Model checkpoint name
    :param steps_per_epochs: Steps to take per epoch?
    :param epochs: Total Epochs to train for
    :param device: Device to train model on
    :param log_interval: Log important params every these many steps
    :param batch_size: Size of batch
    :param seed: Set seed for reproducibility
    :param deterministic_actions: Take deterministic actions during training.
    :type agent: object
    :type logger: object
    :type buffer: object
    :type off_policy: bool
    :type save_interval:int
    :type render: bool
    :type max_ep_len: int
    :type distributed: int
    :type ckpt_log_name: string
    :type steps_per_epochs: int
    :type epochs: int
    :type device: string
    :type log_interval: int
    :type batch_size: int
    :type seed: int
    :type deterministic_actions: bool
    """

    def __init__(
        self,
        agent: Any,
        log_mode: List[str] = ["stdout"],
        save_interval: int = 0,
        render: bool = False,
        max_ep_len: int = 1000,
        distributed: bool = False,
        ckpt_log_name: str = "experiment",
        steps_per_epoch: int = 4000,
        epochs: int = 10,
        device: Union[torch.device, str] = "cpu",
        log_interval: int = 10,
        evaluate_episodes: int = 500,
        logdir: str = "logs",
        batch_size: int = 50,
        seed: Optional[int] = None,
        deterministic_actions: bool = False,
    ):
        super(OnPolicyTrainer, self).__init__(
            agent,
            log_mode,
            buffer=None,
            off_policy=False,
            save_interval=save_interval,
            render=render,
            max_ep_len=max_ep_len,
            distributed=distributed,
            ckpt_log_name=ckpt_log_name,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            device=device,
            log_interval=log_interval,
            evaluate_episodes=evaluate_episodes,
            logdir=logdir,
            batch_size=batch_size,
            seed=seed,
            deterministic_actions=deterministic_actions,
        )

    def train(self) -> None:
        """
        Run training.
        """
        for episode in range(self.epochs):

            epoch_reward = 0

            for i in range(self.agent.actor_batch_size):

                state = self.env.reset()
                done = False

                for t in range(self.agent.timesteps_per_actorbatch):
                    if self.deterministic_actions:
                        action = self.agent.select_action(state, deterministic=True)
                    else:
                        action = self.agent.select_action(state)
                    state, reward, done, _ = self.env.step(np.array(action))

                    if self.render:
                        self.env.render()

                    self.agent.traj_reward.append(reward)

                    if done:
                        break

                epoch_reward += (
                    np.sum(self.agent.traj_reward) / self.agent.actor_batch_size
                )
                self.agent.get_traj_loss()

            if self.agent.__class__.__name__ == "PPO1":
                self.agent.update(
                    episode, episode % self.agent.policy_copy_interval == 0
                )
            else:
                self.agent.update(episode)

            if episode % self.log_interval == 0:
                self.logger.write(
                    {
                        "Episode": episode,
                        "Reward": np.around(epoch_reward, decimals=4),
                        "Timestep": (i * episode * self.agent.timesteps_per_actorbatch),
                    }
                )

            if self.save_interval != 0 and episode % self.save_interval == 0:
                self.checkpoint = self.agent.get_hyperparams()
                save_params(
                    self.agent, i * episode * self.agent.timesteps_per_actorbatch
                )

        self.env.close()
        self.logger.close()
