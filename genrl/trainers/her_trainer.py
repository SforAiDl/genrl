from typing import List, Type, Union

import numpy as np
import torch

from genrl.core import PrioritizedBuffer, ReplayBuffer
from genrl.trainers import OffPolicyTrainer, Trainer
from genrl.utils import safe_mean


class HERTrainer(OffPolicyTrainer):
    def __init__(self, *args, **kwargs):
        super(HERTrainer, self).__init__(*args, **kwargs)

    def _check_state(self, state):
        if isinstance(state, dict):
            return self.env.convert_dict_to_obs(state)
        return state

    def get_action(self, state, timestep):
        if timestep < self.warmup_steps:
            action = self.env.sample()
        else:
            action = self.agent.select_action(self._check_state(state))
        return action

    def check_game_over_status(self, timestep: int, dones: bool):
        game_over = False

        if dones:
            self.training_rewards.append(self.env.episode_reward.detach().clone())
            self.env.reset()
            self.episodes += 1
            game_over = True

        return game_over

    def train(self) -> None:
        """Main training method"""
        if self.load_model is not None:
            self.load()

        state = self.env.reset()
        self.noise_reset()

        self.training_rewards = []
        self.episodes = 0

        for timestep in range(0, self.max_timesteps):
            self.agent.update_params_before_select_action(timestep)

            action = self.get_action(state, timestep)
            next_state, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

            # true_dones contains the "true" value of the dones (game over statuses). It is set
            # to False when the environment is not actually done but instead reaches the max
            # episode length.
            true_dones = info["done"]
            self.buffer.push((state, action, reward, next_state, true_dones))

            state = next_state

            if self.check_game_over_status(timestep, done):
                self.noise_reset()

                if self.episodes % self.log_interval == 0:
                    self.log(timestep)

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
