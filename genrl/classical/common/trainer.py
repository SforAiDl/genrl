import numpy as np
import gym
import matplotlib.pyplot as plt

from genrl.classical.common.models import get_model_from_name


class Trainer:
    """
    Global trainer class for classical RL algorithms
    :param agent: (object) Algorithm object to train
    :param env: (gym environment) standard gym environment to train on
    :param mode: (str) mode of value function update ['learn', 'plan', 'dyna']
    :param model: (str) model to use for planning ['tabular']
    :param n_episodes: (int) number of training episodes
    :param plan_n_steps: (int) number of planning step per environment interaction
    :param start_steps: (int) number of initial exploration timesteps
    :param seed: (int) seed for random number generator
    :param render: (bool) render gym environment
    """

    def __init__(
        self,
        agent,
        env,
        mode="learn",
        model=None,
        n_episodes=30000,
        plan_n_steps=3,
        start_steps=5000,
        start_plan=50,
        seed=None,
        render=False,
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.plan_n_steps = plan_n_steps
        self.start_steps = start_steps
        self.start_plan = start_plan
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
        """
        learn from transition tuples
        :param transitions: (tuple) s, a, r, s' transition
        """
        self.agent.update(transitions)

    def plan(self):
        """
        plans on samples drawn from model
        """
        for i in range(self.plan_n_steps):
            s, a = self.model.sample()
            r, s_ = self.model.step(s, a)
            self.agent.update((s, a, r, s_))

    def train(self):
        """
        general training loop for classical RL
        """
        timestep = 0
        ep = 0
        state = self.env.reset()
        ep_rew = 0
        ep_rews = []

        while True:
            if timestep <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(state)

            next_state, reward, done, _ = self.env.step(action)
            if self.render == True:
                self.env.render()
            ep_rew += reward

            if self.learning == True:
                self.learn((state, action, reward, next_state))

            if self.planning == True and timestep > self.start_plan:
                self.model.add(state, action, reward, next_state)
                if not self.model.is_empty():
                    self.plan()

            state = next_state
            if done == True:
                ep_rews.append(ep_rew)
                if ep % 100 == 0:
                    print(
                        "Episode: {}, Reward: {}, timestep: {}".format(
                            ep, ep_rew, timestep
                        )
                    )

                if ep == self.n_episodes:
                    break

                ep += 1
                state = self.env.reset()
                ep_rew = 0

            timestep += 1
        self.env.close()

        return ep_rews

    def plot(self, results, window_size=100):
        """
        plot model rewards
        :param results: (list) rewards for each episode
        :param window_size: (int) size of moving average filter
        """
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
        agent, env, mode="dyna", model="tabular", seed=42, n_episodes=50, start_steps=0
    )
    ep_rs = trainer.train()
    trainer.plot(ep_rs)
