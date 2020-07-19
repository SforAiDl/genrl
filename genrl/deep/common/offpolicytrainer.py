import os
from abc import ABC
from typing import Any, List, Optional, Type, Union

import gym
import numpy as np
import torch

from ...environments import VecEnv
from .buffers import PrioritizedBuffer, ReplayBuffer
from .logger import Logger
from .trainer import Trainer
from .utils import safe_mean, set_seeds


class OffPolicyTrainer(Trainer):
    def __init__(
        self,
        agent: Any,
        env: Union[gym.Env, VecEnv],
        log_mode: List[str] = ["stdout"],
        buffer: Union[Type[ReplayBuffer], Type[PrioritizedBuffer]] = None,
        off_policy: bool = True,
        save_interval: int = 0,
        save_model: str = "checkpoints",
        run_num: int = None,
        load_model: str = None,
        render: bool = False,
        max_ep_len: int = 1000,
        distributed: bool = False,
        steps_per_epoch: int = 200,
        epochs: int = 25,
        device: Union[torch.device, str] = "cpu",
        log_interval: int = 10,
        evaluate_episodes: int = 500,
        logdir: str = "logs",
        batch_size: int = 50,
        seed: Optional[int] = 0,
        deterministic_actions: bool = False,
        warmup_steps: int = 1000,
        start_update: int = 1000,
        update_interval: int = 50,
    ):
        super(OffPolicyTrainer, self).__init__(
            agent,
            env,
            log_mode=log_mode,
            buffer=buffer,
            off_policy=off_policy,
            save_interval=save_interval,
            save_model=save_model,
            run_num=run_num,
            render=render,
            max_ep_len=max_ep_len,
            distributed=distributed,
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
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.start_update = start_update
        self.network_type = self.agent.network_type

    def train(self) -> None:
        if self.load_mode is not None:
            self.load()

        state = self.env.reset()

        for epoch in range(self.epochs):
            for timestep in range(self.steps_per_epoch):
                self.agent.update_params_before_select_action(timestep * epoch)

                if timestep < self.warmup_steps:
                    action = np.array(self.env.sample())
                else:
                    action = self.agent.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                if self.render:
                    self.env.render()
