import numpy as np


class QLearning:
    """
    Q-Learning Algorithm
    
    Paper- https://link.springer.com/article/10.1007/BF00992698

    :param env: standard gym environment to train on
    :param epsilon: exploration coefficient
    :param gamma: discount factor
    :param lr: learning rate
    :type env: Gym environment 
    :type epsilon: float
    :type gamma: float
    :type lr: float
    """

    def __init__(self, env, epsilon=0.9, gamma=0.95, lr=0.01):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def get_action(self, s, explore=True):
        '''
        Epsilon greedy selection of epsilon in the explore phase 

        :param s: Current state
        :param explore: Whether you are exploring or exploiting
        :type s: int, float, ...
        :type explore: bool 
        :returns: Action based on the Q table 
        :rtype: int, float, ...  
        '''
        if explore == True:
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
        return np.argmax(self.Q[s, :])

    def update(self, transition):
        '''
        Update the Q table

        :param transition: step taken in the envrionment 
        '''
        s, a, r, s_ = transition

        self.Q[s, a] += self.lr * (
            r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a]
        )
