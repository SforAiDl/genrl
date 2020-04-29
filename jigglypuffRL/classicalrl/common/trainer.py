import numpy as np
import gym
import matplotlib.pyplot as plt

from jigglypuffRL.classicalrl.common.models import get_model_from_name
from jigglypuffRL.classicalrl import (
    QLearning
)

class Trainer:
    def __init__(self, agent, env, mode='learn', model=None, n_episodes=30000, plan_n_steps=50, start_steps=10000, seed=None):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.plan_n_steps = plan_n_steps
        self.start_steps = start_steps

        if mode == 'learn':
            self.learning = True
            self.planning = False
        elif mode == 'plan':
            self.learning = False
            self.planning = True
        elif mode == 'dyna':
            self.learning = True
            self.planning = True

        if seed is not None:
            np.random.seed(seed)

        if self.planning == True and model is not None:
            self.model = get_model_from_name(model)()

    def learn(self, transitions):
        self.agent.update(transitions)

    def plan(self, n_steps):
        for i in range(n_steps):
            s, a = self.model.sample()
            r, s_ = self.model.step(s, a)
            self.agent.update((s, a, r, s_))

    def train(self):
        t = 0
        ep = 0
        s = self.env.reset()
        ep_r = 0
        ep_rs = []

        while True:
            if t <= self.start_steps:
                a = self.env.action_space.sample()
            else:
                a = self.agent.get_action(s)
            s_, r, done, _ = env.step(a)
            ep_r += r
            
            if self.learning == True:
                self.learn((s, a, r, s_))

            if self.planning == True:
                self.model.add(s, a, r, s_)
                self.plan(self.plan_n_steps)

            s = s_
            if done == True:
                ep_rs.append(ep_r)
                if ep % 100 == 0:
                    print('Episode: {}, Reward: {}, timestep: {}'.format(ep, ep_r, t))
                
                if ep == self.n_episodes:
                    break

                ep += 1
                s = self.env.reset()
                ep_r = 0
                
            t += 1
        return ep_rs

    def plot(self, results, window_size=100):
        avgd_results = [0] * len(results)
        for i in range(window_size, len(results)):
            avgd_results[i] = np.mean(results[i-window_size:i])
        
        plt.plot(list(range(0, len(results))), avgd_results)
        plt.title('Results')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QLearning(env)
    trainer = Trainer(agent, env, seed=42, n_episodes=50000)
    ep_rs = trainer.train()
    trainer.plot(ep_rs)
