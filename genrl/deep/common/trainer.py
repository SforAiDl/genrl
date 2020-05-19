import os

import torch
from torchvision import transforms
import numpy as np
from collections import deque

from abc import ABC

from .utils import set_seeds
from .logger import Logger


class Trainer(ABC):
    """
    Base Trainer class. To be inherited specific usecases.

    :param agent: Algorithm object
    :param env: Standard gym environment
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
    :type env: object
    :type logger: object
    :type buffer: object
    :type off_policy: bool
    :type save_interval: int
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
        agent,
        env,
        log_mode=["stdout"],
        buffer=None,
        off_policy=False,
        save_interval=0,
        render=False,
        max_ep_len=1000,
        distributed=False,
        ckpt_log_name="experiment",
        steps_per_epoch=4000,
        epochs=10,
        device="cpu",
        log_interval=10,
        evaluate_episodes=500,
        logdir="logs",
        batch_size=50,
        seed=None,
        deterministic_actions=False,
        transform=None,
        history_length=4,
    ):
        self.agent = agent
        self.env = env
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

    def train(self):
        """
        To be defined in inherited classes
        """
        raise NotImplementedError

    def save(self):
        """
        Save function. It calls `get_hyperparams` method of agent to \
get important model hyperparams.
        Creates a checkpoint `{logger_dir}/{algo}_{env_name}/{ckpt_log_name}
        """
        saving_params = self.agent.get_hyperparams()
        logdir = self.logger.logdir
        algo = self.agent.__class__.__name__
        env_name = self.env.envs[0].unwrapped.spec.id

        save_dir = "{}/checkpoints/{}_{}".format(logdir, algo, env_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(saving_params, "{}/{}.pt".format(save_dir, self.ckpt_log_name))

    def evaluate(self):
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
                    print("Evaluated for {} episodes, Mean Reward: {}, Std Deviation for the Reward: {}".format(
                        self.evaluate_episodes, np.mean(ep_rews), np.std(ep_rews)
                    ))
                    break

    @property
    def n_envs(self):
        """
        Number of environments
        """
        return self.env.n_envs


class OffPolicyTrainer(Trainer):
    """
    Off-Policy Trainer class

    :param agent: Algorithm object
    :param env: Standard gym environment
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
    :type env: object
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
        agent,
        env,
        log_mode=["stdout"],
        buffer=None,
        off_policy=True,
        save_interval=0,
        render=False,
        max_ep_len=1000,
        distributed=False,
        ckpt_log_name="experiment",
        steps_per_epoch=4000,
        epochs=10,
        device="cpu",
        log_interval=10,
        evaluate_episodes=500,
        logdir="logs",
        batch_size=50,
        seed=0,
        deterministic_actions=False,
        warmup_steps=10000,
        start_update=1000,
        update_interval=50,
    ):
        super(OffPolicyTrainer, self).__init__(
            agent,
            env,
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

        if self.network_type == "cnn":
            if self.transform is None:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Grayscale(),
                        transforms.Resize((110, 84)),
                        transforms.CenterCrop(84),
                        transforms.ToTensor(),
                    ]
                )

            self.state_history = deque(
                [
                    self.transform(self.env.observation_space.sample())
                    for _ in range(self.history_length)
                ],
                maxlen=self.history_length,
            )

    def train(self):
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

            if self.network_type == "cnn":
                self.state_history.append(self.transform(state))
                phi_state = torch.stack(list(self.state_history), dim=1)

        for t in range(total_steps):
            if self.agent.__class__.__name__ == "DQN":
                self.agent.epsilon = self.agent.calculate_epsilon_by_frame(t)

                if self.network_type == "cnn":
                    action = self.agent.select_action(phi_state)
                else:
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

            if self.agent.__class__.__name__ == "DQN" and self.network_type == "cnn":
                self.state_history.append(self.transform(next_state))
                phi_next_state = torch.stack(list(self.state_history), dim=1)
                self.buffer.push((phi_state, action, reward, phi_next_state, done))
                phi_state = phi_next_state
            else:
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
                            "Episode Reward": episode_reward,
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
                self.save()

        self.env.close()
        self.logger.close()


class OnPolicyTrainer(Trainer):
    """
    Base Trainer class. To be inherited specific usecases.

    :param agent: Algorithm object
    :param env: Standard gym environment
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
    :type env: object
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
        agent,
        env,
        log_mode=["stdout"],
        save_interval=0,
        render=False,
        max_ep_len=1000,
        distributed=False,
        ckpt_log_name="experiment",
        steps_per_epoch=4000,
        epochs=10,
        device="cpu",
        log_interval=10,
        evaluate_episodes=500,
        logdir="logs",
        batch_size=50,
        seed=None,
        deterministic_actions=False,
    ):
        super(OnPolicyTrainer, self).__init__(
            agent,
            env,
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

    def train(self):
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
                        "Reward": epoch_reward,
                        "Timestep": (i * episode * self.agent.timesteps_per_actorbatch),
                    }
                )

            if self.save_interval != 0 and episode % self.save_interval == 0:
                self.checkpoint = self.agent.get_hyperparams()
                self.save()

        self.env.close()
        self.logger.close()
