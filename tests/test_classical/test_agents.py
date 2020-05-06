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
        assert np.all(agent.Q == answer)

        assert agent.get_action(3,False) == 1