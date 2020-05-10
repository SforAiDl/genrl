import numpy as np


class QLearning:
    """
    Q-Learning Algorithm
    Paper: https://link.springer.com/article/10.1007/BF00992698

    :param env: Standard gym environment to train on
    :param epsilon: Exploration coefficient
    :param gamma: Discount factor
    :param lr: Learning rate
    :type env: Gym Environment
    :type epsilon: float
    :type gamma: float
    :type lr: float
    """
    def __init__(self, env, epsilon=0.9, gamma=0.95, lr=0.01):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros((
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

        self.Q[state, action] += self.lr * (
            reward
            + self.gamma * np.max(self.Q[next_state, :])
            - self.Q[state, action]
        )
