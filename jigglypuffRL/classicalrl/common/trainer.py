import numpy as np
import gym
import matplotlib.pyplot as plt

from jigglypuffRL.classicalrl.common.models import get_model_from_name


class Trainer:
    def __init__(
        self,
        agent,
        env,
        mode="learn",
        model=None,
        n_episodes=30000,
        plan_n_steps=3,
        start_steps=5000,
        seed=None,
        render=False,
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.plan_n_steps = plan_n_steps
        self.start_steps = start_steps
        self.render = render

        if mode == "learn":
            self.learning = True
            self.planning = False
        elif mode == "plan":
            self.learning = False
            self.planning = True
        elif mode == "dyna":
            self.learning = True
            self.planning = True

        if seed is not None:
            np.random.seed(seed)

        if self.planning == True and model is not None:
            self.model = get_model_from_name(model)(
                self.env.observation_space.n, self.env.action_space.n
            )

    def learn(self, transitions):
        self.agent.update(transitions)

    def plan(self):
        for i in range(self.plan_n_steps):
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
            if self.render == True:
                self.env.render()
            ep_r += r

            if self.learning == True:
                self.learn((s, a, r, s_))

            if self.planning == True:
                self.model.add(s, a, r, s_)
                self.plan()

            s = s_
            if done == True:
                ep_rs.append(ep_r)
                if ep % 100 == 0:
                    print("Episode: {}, Reward: {}, timestep: {}".format(ep, ep_r, t))

                if ep == self.n_episodes:
                    break

                ep += 1
                s = self.env.reset()
                ep_r = 0

            t += 1
        self.env.close()

        return ep_rs

    def plot(self, results, window_size=100):
        avgd_results = [0] * len(results)
        for i in range(window_size, len(results)):
            avgd_results[i] = np.mean(results[i - window_size : i])

        plt.plot(list(range(0, len(results))), avgd_results)
        plt.title("Results")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = QLearning(env)
    trainer = Trainer(
        agent, env, mode="dyna", model="tabular", seed=42, n_episodes=10000
    )
    ep_rs = trainer.train()
    trainer.plot(ep_rs)
