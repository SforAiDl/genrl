import gym
import numpy as np
import pytest

from genrl import SARSA, QLearning
from genrl.classical.common import Trainer


class TestTrainer:
    def test_trainer(self):
        env = gym.make("FrozenLake-v0")
        agent = QLearning(env)
        trainer = Trainer(
            agent,
            env,
            mode="dyna",
            model="tabular",
            n_episodes=50,
            start_steps=0,
            evaluate_frequency=1,
        )
        ep_rs = trainer.train()
        trainer.evaluate()
