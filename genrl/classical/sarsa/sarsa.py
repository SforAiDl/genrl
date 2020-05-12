import numpy as np


class SARSA:
    """
    SARSA Algorithm
    Paper- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.2539&rep=rep1&type=pdf

    :param env: Standard gym environment to train on
    :param epsilon: Exploration coefficient
    :param lmbda: Eligibility trace coefficient
    :param gamma: Discount factor
    :param lr: Learning rate
    :type env: Gym Environment
    :type epsilon: float
    :type lmbda: float
    :type gamma: float
    :type lr: float
    """
    def __init__(self, env, epsilon=0.9, lmbda=0.9, gamma=0.95, lr=0.01):
        self.env = env
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros((
            self.env.observation_space.n, self.env.action_space.n
        ))

        self.e = np.zeros((
            self.env.observation_space.n, self.env.action_space.n
        ))

    def get_action(self, state, explore=True):
        """
        Gives action which would give the highest Q-value

        :param state: Input state for finding Q-values
        :param explore: True if agent is exploring, else False
        :type state: Numpy array
        :type explore: bool
        :returns: Action with the highest Q-value for the given state
        """
        if explore is True:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[state, :])

    def update(self, transition):
        """
        Updates Q-values given next transition tuple

        :param transition: Tuple of state, action, reward and next state
        :type transition: tuple
        """
        state, action, reward, next_state = transition

        next_action = self.get_action(next_state)
        delta = (
            reward
            + self.gamma * self.Q[next_state, next_action]
            - self.Q[state, action]
        )
        self.e[state, action] += 1

        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                self.Q[state, action] += (
                    self.lr * delta * self.e[state, action]
                )
                self.e[state, action] = (
                    self.gamma * self.lmbda * self.e[state, action]
                )
