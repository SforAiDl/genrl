import gym
import numpy as np


class ValueIterator:
    def __init__(self, env, epsilon=0.9, gamma=0.95):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.V = np.zeros(self.env.observation_space.n)
        self.dynamics_model = self.env.P

    def get_action(self, state, explore=True):
        """

        Args:
            state: The state the agent is currently in
            explore (bool): True is agent is allowed to explore, else False

        Returns:

        """
        if explore:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        Qs = self._lookahead(state)
        best_action = np.argmax(Qs)
        return best_action

    def update(self, transition):
        """
        One Step of value iteration
        Args:
            transition (Tuple): Containing the state, action, reward, next_state

        Returns:

        """
        for s in range(self.env.observation_space.n):
            Qs = self._lookahead(s)
            best_action = np.argmax(Qs)
            self.V[s] = Qs[best_action]

    def _lookahead(self, state):
        Qs = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for i in range(len(self.dynamics_model[state][a])):
                prob, next_state, reward, done = self.dynamics_model[state][a][i]
                Qs[a] += prob * (reward + self.gamma * self.V[next_state])
        return Qs
