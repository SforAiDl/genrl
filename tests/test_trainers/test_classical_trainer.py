import gym
import numpy as np
import pytest

from genrl.agents import SARSA, QLearning
from genrl.trainers import ClassicalTrainer


class TestClassicalTrainer:
    def test_classical_trainer(self):
        env = gym.make("FrozenLake-v0")
        agent = QLearning(env)
        trainer = ClassicalTrainer(
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
