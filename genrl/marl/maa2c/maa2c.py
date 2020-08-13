import gc

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from a2c_agent import A2CAgent
from logger import TensorboardLogger
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from trainer import OnPolicyTrainer


class MAA2C:
    def __init__(self, env, max_steps, max_episodes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.num_agents = env.n
        self.agents = A2CAgent(self.env)

        self.agents.steps_per_episode = 300

        self.trainer = OnPolicyTrainer(
            agent=self.agents,
            env=self.env,
            save_interval=500,
            steps_per_epoch=max_steps,
            epochs=max_episodes,
            device=self.device,
            log_interval=1,
            logdir="logs/",
        )
        self.trainer.train()
