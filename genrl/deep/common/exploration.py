# create decorator functions for exploration methods
# decorator takes as param the select_action method and adds noise to it
# Some exploration methods require networks/access to the model
import numpy as np

import gym

class EpsilonGreedyDecorator:
    def __init__(self, env, epsilon):
        self.env = env
        self.epsilon = epsilon

    def __call__(self, select_action_func):
        def inner(*args, **kwargs):
            if np.random.uniform() > self.epsilon:
                return self.env.action_space.sample()
            else:
                return select_action_func(*args, **kwargs)
        return inner

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    
    def add(a, b):
        return a + b

    for _ in range(50):
        print((EpsilonGreedyDecorator(env, 1.0))(add)(1,2))