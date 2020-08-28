from typing import Type, Union

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
        warmup_steps (int): Number of warmup steps. (random actions are taken to add randomness to training)
        start_update (int): Timesteps after which the agent networks should start updating
        update_interval (int): Timesteps between target network updates
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
        *args,
        buffer: Union[Type[ReplayBuffer], Type[PrioritizedBuffer]] = None,
        max_ep_len: int = 500,
        warmup_steps: int = 1000,
        start_update: int = 1000,
        update_interval: int = 50,
        **kwargs
    ):
        super(OffPolicyTrainer, self).__init__(*args, off_policy=True, **kwargs)

        self.max_ep_len = max_ep_len
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.start_update = start_update
        self.network = self.agent.network

        if buffer is None:
            if self.agent.replay_buffer is None:
                raise Exception("Off Policy Training requires a Replay Buffer")
        else:
            self.agent.replay_buffer = buffer
        self.buffer = self.agent.replay_buffer

    def train(self) -> None:
        """Main training method"""
        if self.load_model is not None:
            self.load()

        state, episode_len, episode = (
            self.env.reset(),
            np.zeros(self.env.n_envs),
            np.zeros(self.env.n_envs),
        )
        total_steps = self.max_ep_len * self.epochs * self.env.n_envs

        if "noise" in self.agent.__dict__ and self.agent.noise is not None:
            self.agent.noise.reset()

        assert self.update_interval % self.env.n_envs == 0

        self.rewards = []

        for timestep in range(0, total_steps, self.env.n_envs):
            self.agent.update_params_before_select_action(timestep)

            if timestep < self.warmup_steps:
                action = np.array(self.env.sample())
            else:
                action = self.agent.select_action(state)

            next_state, reward, done, _ = self.env.step(action)

            if self.render:
                self.env.render()

            episode_len += 1

            done = [
                False if episode_len[i] == self.max_ep_len else done[i]
                for i, ep_len in enumerate(episode_len)
            ]

            self.buffer.push((state, action, reward, next_state, done))
            state = next_state.copy()

            if np.any(done) or np.any(episode_len == self.max_ep_len):
                if "noise" in self.agent.__dict__ and self.agent.noise is not None:
                    self.agent.noise.reset()

                if sum(episode) % self.log_interval == 0:
                    self.logger.write(
                        {
                            "timestep": timestep,
                            "Episode": sum(episode),
                            **self.agent.get_logging_params(),
                            "Episode Reward": safe_mean(self.rewards),
                        },
                        self.log_key,
                    )
                    self.rewards = []

                for i, di in enumerate(done):
                    if di:
                        self.rewards.append(self.env.episode_reward[i])
                        self.env.reset_single_env(i)
                        episode_len[i] = 0
                        episode[i] += 1

            if timestep >= self.start_update and timestep % self.update_interval == 0:
                self.agent.update_params(self.update_interval)

            if (
                timestep >= self.start_update
                and self.save_interval != 0
                and timestep % self.save_interval == 0
            ):
                self.save(timestep)

            if self.max_timesteps is not None and timestep >= self.max_timesteps:
                break

        self.env.close()
        self.logger.close()
