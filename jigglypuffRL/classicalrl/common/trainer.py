import numpy as np

from jigglypuffRL.classicalrl.common.models import get_model_from_name

class Trainer:
    def __init__(self, agent, env, mode, model, seed):
        self.agent = agent
        self.env = env
        self.mode = mode
        self.model = model

        if seed is not None:
            np.random.seed(seed)

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
    def __init__(self, agent, env, mode='learn', model=None, n_episodes=100, batch_size=50, seed=None):
        super(OnPolicyTrainer, self).__init__(agent, env, mode, model, seed)

        if model is not None:
            self.model = get_model_from_name(model)
        
        self.n_episodes = n_episodes
        self.batch_size = batch_size

    def train(self):
        t = 0
        s = self.env.reset()

        while True:
            if self.select_action()