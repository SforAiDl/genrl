import pytest
import numpy as np
import gym

from genrl import QLearning, SARSA

class TestAgents:
    def test_qlearning(self):
        env = gym.make('FrozenLake-v0')
        agent = QLearning(env)

        assert np.any(agent.Q) == False

        agent.update((3, 1, 3.1, 4))
        answer = np.zeros((16,4))
        answer[3,1] += 0.031000000000000003
        assert np.all(agent.Q == pytest.approx(answer))

        assert agent.get_action(3,False) == 1

    def test_sarsa(self):
        env = gym.make('FrozenLake-v0')
        agent = SARSA(env)

        assert np.any(agent.Q) == False
        assert np.any(agent.e) == False

        agent.update((3, 1, 3.1, 4))
        answer_Q = np.zeros((16,4))
        answer_e = np.zeros((16,4))
        answer_Q[3,1] = 0.21378364
        answer_e[3,1] = 4.42416527e-05
        
        assert np.all(agent.Q == pytest.approx(answer_Q))
        assert np.all(agent.e == pytest.approx(answer_e))
        