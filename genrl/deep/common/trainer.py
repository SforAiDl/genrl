import os

import gym
import torch
import numpy as np

from genrl.deep.common import set_seeds, Logger, venv
# from genrl import SAC
from abc import ABC


class Trainer(ABC):
    """
    Base Trainer class. To be inherited specific usecases.
    :param agent: (object) Algorithm object
    :param env: (object) standard gym environment
    :param logger: (object) Logger object
    :param buffer: (object) Buffer Object
    :param off_policy: (bool) Is the algorithm off-policy?
    :param save_interval:(int) Model to save in each of these many timesteps
    :param render: (bool) Should the Environment render
    :param max_ep_len: (int) Max Episode Length
    :param distributed: (int) Should distributed training be enabled? (To be implemented)
    :param ckpt_log_name: (string) Model checkpoint name
    :param steps_per_epochs: (int) Steps to take per epoch?
    :param epochs: (int) Total Epochs to train for
    :param device: (string) Device to train model on
    :param log_interval: (int) Log important params every these many steps
    :param batch_size: (int) Size of batch
    :param seed: (int) Set seed for reproducibility
    :param deterministic_actions: (bool) Take deterministic actions during training.
    """

    def __init__(
        self,
        agent,
        env,
        logger,
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
        batch_size=50,
        seed=None,
        deterministic_actions=False,
    ):
        self.agent = agent
        self.env = env
        self.logger = logger
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
        self.batch_size = batch_size
        self.determinsitic_actions = deterministic_actions
        if seed is not None:
            set_seeds(seed, self.env)

    def train(self):
        """
        To be defined in inherited classes
        """
        raise NotImplementedError

    def save(self):
        """
        Save function. It calls `get_hyperparams` method of agent to get important model hyperparams.
        Creates a checkpoint `{logger_dir}/{algo}_{env_name}/{ckpt_log_name}
        """
        saving_params = self.agent.get_hyperparams()
        logdir = self.logger.logdir
        algo = self.agent.__class__.__name__
        env_name = self.env.envs[0].unwrapped.spec.id

        save_dir = "{}/checkpoints/{}_{}".format(logdir, algo, env_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(saving_params, "{}/{}.pt".format(save_dir, self.ckpt_log_name))

    @property
    def n_envs(self):
        return self.env.n_envs


class OffPolicyTrainer(Trainer):
    """
    Off-Policy Trainer class. 
    :param agent: (object) Algorithm object
    :param env: (object) standard gym environment
    :param logger: (object) Logger object
    :param buffer: (object) Buffer Object. Cannot be None for Off-policy
    :param off_policy: (bool) Is the algorithm off-policy?
    :param save_interval:(int) Model to save in each of these many timesteps
    :param render: (bool) Should the Environment render
    :param max_ep_len: (int) Max Episode Length
    :param distributed: (int) Should distributed training be enabled? (To be implemented)
    :param ckpt_log_name: (string) Model checkpoint name
    :param steps_per_epochs: (int) Steps to take per epoch?
    :param epochs: (int) Total Epochs to train for
    :param device: (string) Device to train model on
    :param log_interval: (int) Log important params every these many steps
    :param batch_size: (int) Size of batch
    :param seed: (int) Set seed for reproducibility
    :param deterministic_actions: (bool) Take deterministic actions during training.
    :param warmup_steps: (int) Observe the environment for these many steps with randomly sampled actions to store in buffer.
    :param start_update: (int) Starting updating the policy after these many steps
    :param update_interval: (int) Update model policies after number of steps.
    """

    def __init__(
        self,
        agent,
        env,
        logger,
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
        batch_size=50,
        seed=0,
        deterministic_actions=False,
        warmup_steps=10000,
        start_update=10000,
        update_interval=50,
    ):
        super(OffPolicyTrainer, self).__init__(
            agent,
            env,
            logger,
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
            batch_size,
            seed,
            deterministic_actions,
        )
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.start_update = start_update

    def train(self):
        """
        Run training
        """
        state, episode_reward, episode_len, episode = self.env.reset(), np.zeros(self.env.n_envs), np.zeros(self.env.n_envs), np.zeros(self.env.n_envs)
        total_steps = self.steps_per_epoch * self.epochs * self.env.n_envs
        # self.agent.learn()

        if "noise" in self.agent.__dict__ and self.agent.noise is not None:
            self.agent.noise.reset()

        if self.agent.__class__.__name__ == "DQN":
            self.agent.update_target_model()

        assert self.update_interval%self.env.n_envs==0

        for t in range(0, total_steps, self.env.n_envs):
            if self.agent.__class__.__name__ == "DQN":
                self.agent.epsilon = self.agent.calculate_epsilon_by_frame(t)
                action = self.agent.select_action(state)

            else:
                if t < self.warmup_steps:
                    action = np.array(self.env.sample())
                    # print("Observing")
                    # print(len(action))
                else:
                    if self.determinsitic_actions:
                        action = self.agent.select_action(state, deterministic=True)
                    else:
                        action = self.agent.select_action(state)                    

            next_state, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

            episode_reward += reward
            episode_len += 1

            done = [False if episode_len[i]==self.max_ep_len else done[i] for i, ep_len in enumerate(episode_len)]

            # print("s", state, "ns", next_state)

            self.buffer.extend(zip(state, action, reward, next_state, done))

            state = next_state.copy()

            if np.any(done) or np.any(episode_len == self.max_ep_len):
                if "noise" in self.agent.__dict__ and self.agent.noise is not None:
                    self.agent.noise.reset()

                if sum(episode) % self.log_interval == 0:
                    self.logger.write(
                        {
                            "timestep": t,
                            "Episode": sum(episode),
                            "Episode Reward": np.mean(episode_reward),
                        }
                    )

                for i, d in enumerate(done):
                    if d:
                        episode_reward[i] = 0
                        episode_len[i] = 0
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
                        states, actions, rewards, next_states, dones = (
                            x.to(self.device) for x in batch
                        )
                        if self.agent.__class__.__name__ == "TD3":
                            self.agent.update_params(
                                states, actions, rewards, next_states, dones, _
                            )
                        elif self.agent.__class__.__name__ == "DDPG":
                            self.agent.update_params(
                                states, actions, rewards, next_states, dones
                            )
                        else:
                            self.agent.update_params(
                                states, actions, rewards, next_states, dones
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
    :param agent: (object) Algorithm object
    :param env: (object) standard gym environment
    :param logger: (object) Logger object
    :param buffer: (object) Buffer Object
    :param off_policy: (bool) Is the algorithm off-policy?
    :param save_interval:(int) Model to save in each of these many timesteps
    :param render: (bool) Should the Environment render
    :param max_ep_len: (int) Max Episode Length
    :param distributed: (int) Should distributed training be enabled? (To be implemented)
    :param ckpt_log_name: (string) Model checkpoint name
    :param steps_per_epochs: (int) Steps to take per epoch?
    :param epochs: (int) Total Epochs to train for
    :param device: (string) Device to train model on
    :param log_interval: (int) Log important params every these many steps
    :param batch_size: (int) Size of batch
    :param seed: (int) Set seed for reproducibility
    :param deterministic_actions: (bool) Take deterministic actions during training.
    """

    def __init__(
        self,
        agent,
        env,
        logger,
        save_interval=0,
        render=False,
        max_ep_len=1000,
        distributed=False,
        ckpt_log_name="experiment",
        steps_per_epoch=4000,
        epochs=10,
        device="cpu",
        log_interval=10,
        batch_size=50,
        seed=None,
        deterministic_actions=False,
    ):
        super().__init__(
            agent,
            env,
            logger,
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
                    if self.determinsitic_actions:
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

            self.agent.update_policy(
                episode, episode % self.agent.policy_copy_interval == 0
            )

            if episode % self.log_interval == 0:
                self.logger.write(
                    {
                        "Episode": episode,
                        "Reward": epoch_reward,
                        "Timestep": i * episode * self.agent.timesteps_per_actorbatch,
                    }
                )

            if self.save_interval != 0 and episode % self.save_interval == 0:
                self.checkpoint = self.agent.get_hyperparams()
                self.save()

            self.env.close()
            self.logger.close()


if __name__ == "__main__":
    log_dir = os.getcwd()
    logger = Logger(log_dir, ["stdout"])
    # env = gym.make("Pendulum-v0")
    env = venv("Pendulum-v0", 16, parallel=False)
    algo = SAC("mlp", env, seed=0)

    import time

    start = time.time()
    trainer = OffPolicyTrainer(algo, env, logger, render=True, seed=0, epochs=10)
    trainer.train()
    end = time.time()

    print(end - start)
    # algo = VPG("mlp", env, seed=0)
    # trainer = OnPolicyTrainer(algo, env, logger, render=True, seed=0, epochs=100, log_interval=1)
    # trainer.train()
