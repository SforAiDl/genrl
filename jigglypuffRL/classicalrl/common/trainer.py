import numpy as np

from jigglypuffRL.classicalrl.common.models import get_model_from_name

class Trainer:
    def __init__(self, agent, env, mode, model, seed):
        self.agent = agent
        self.env = env
        self.model = model

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

        self.s_dim = self.env.observation_space.n
        self.a_dim = self.env.action_space.n

    def learn(self, transitions):
        self.agent.update(transitions)

    def plan(self, n_steps):
        for i in range(n_steps):
            s, a = self.model.sample()
            r, s_ = self.model.step(s, a)
            self.agent.update((s, a, r, s_))

    def train(self):
        raise NotImplementedError        


class OnPolicyTrainer(Trainer):
    def __init__(self, agent, env, mode='learn', model=None, n_episodes=100, batch_size=50, plan_n_steps=50, seed=None):
        super(OnPolicyTrainer, self).__init__(agent, env, mode, model, seed)

        if self.planning == True and model is not None:
            self.model = get_model_from_name(model)()
        
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.plan_n_steps = plan_n_steps

    def train(self):
        t = 0
        s = self.env.reset()

        while True:
            a = self.agent.get_action()
            s_, r, done, _ = env.step(a)
            
            if self.learning == True:
                self.learn((s, a, r, s_))

            if self.planning == True:
                self.model.add(s, a, r, s_)
                self.plan(self.plan_n_steps)

            s = s_
            if done == True:
                s = self.env.reset()
                
            t += 1