import gym
import numpy as np
import pytest

from genrl.agents import SARSA, QLearning, ValueIterator
from genrl.trainers import ClassicalTrainer


class TestClassicalAgents:
    def test_qlearning(self):
        env = gym.make("FrozenLake-v0")
        agent = QLearning(env)

        assert not np.any(agent.Q)

        agent.update((3, 1, 3.1, 4))
        answer = np.zeros((16, 4))
        answer[3, 1] += 0.031000000000000003
        assert np.all(agent.Q == pytest.approx(answer))

        assert agent.get_action(3, False) == 1

    def test_sarsa(self):
        env = gym.make("FrozenLake-v0")
        agent = SARSA(env)

        assert not np.any(agent.Q)
        assert not np.any(agent.e)

        agent.update((3, 1, 3.1, 4))
        answer_Q = np.zeros((16, 4))
        answer_e = np.zeros((16, 4))
        answer_Q[3, 1] = 0.21378364
        answer_e[3, 1] = 4.42416527e-05

        assert np.all(agent.Q == pytest.approx(answer_Q))
        assert np.all(agent.e == pytest.approx(answer_e))

    def test_valueiteration(self):
        env = gym.make("FrozenLake-v0")
        agent = ValueIterator(env)
        trainer = ClassicalTrainer(
            agent,
            env,
            mode="dyna",
            model="tabular",
            n_episodes=5,
            start_steps=0,
            evaluate_frequency=1,
        )
        _ = trainer.train()
        trainer.evaluate()
