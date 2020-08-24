from typing import Any, List, Optional, Tuple

import gym
import numpy as np
from matplotlib import pyplot as plt

from genrl.classical.common.models import get_model_from_name


class Trainer:
    """
    Global trainer class for classical RL algorithms

    :param agent: Algorithm object to train
    :param env: standard gym environment to train on
    :param mode: mode of value function update ['learn', 'plan', 'dyna']
    :param model: model to use for planning ['tabular']
    :param n_episodes: number of training episodes
    :param plan_n_steps: number of planning step per environment interaction
    :param start_steps: number of initial exploration timesteps
    :param seed: seed for random number generator
    :param render: render gym environment
    :type agent: object
    :type env: Gym environment
    :type mode: str
    :type model: str
    :type n_episodes: int
    :type plan_n_steps: int
    :type start_steps: int
    :type seed: int
    :type render: bool
    """

    def __init__(
        self,
        agent: Any,
        env: gym.Env,
        mode: str = "learn",
        model: str = None,
        n_episodes: int = 30000,
        plan_n_steps: int = 3,
        start_steps: int = 5000,
        start_plan: int = 50,
        evaluate_frequency: int = 500,
        seed: Optional[int] = None,
        render: bool = False,
    ):
        self.agent = agent
        self.env = env
        self.n_episodes = n_episodes
        self.plan_n_steps = plan_n_steps
        self.start_steps = start_steps
        self.start_plan = start_plan
        self.evaluate_frequency = evaluate_frequency
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

        if self.planning is True and model is not None:
            self.model = get_model_from_name(model)(
                self.env.observation_space.n, self.env.action_space.n
            )

    def learn(self, transitions: Tuple) -> None:
        """
        learn from transition tuples

        :param transitions: s, a, r, s' transition
        :type transitions: tuple
        """
        self.agent.update(transitions)

    def plan(self) -> None:
        """
        plans on samples drawn from model
        """
        for _ in range(self.plan_n_steps):
            state, action = self.model.sample()
            reward, next_state = self.model.step(state, action)
            self.agent.update((state, action, reward, next_state))

    def train(self) -> List[float]:
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
            if self.render:
                self.env.render()
            ep_rew += reward

            if self.learning:
                self.learn((state, action, reward, next_state))

            if self.planning is True and timestep > self.start_plan:
                self.model.add(state, action, reward, next_state)
                if not self.model.is_empty():
                    self.plan()

            state = next_state
            if done:
                ep_rews.append(ep_rew)
                if ep % self.evaluate_frequency == 0:
                    print("Evaluating at the episode number: {}".format(ep))
                    self.evaluate()

                if ep == self.n_episodes:
                    break

                ep += 1
                state = self.env.reset()
                ep_rew = 0

            timestep += 1
        self.env.close()

        return ep_rews

    def evaluate(self, eval_ep: int = 100) -> None:
        """
        Evaluate function.

        :param eval_ep: Number of episodes you want to evaluate for
        :type eval_ep: int
        """
        ep = 0
        ep_rew = 0
        ep_rews = []
        state = self.env.reset()

        while True:
            action = self.agent.get_action(state, False)
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            ep_rew += reward
            if done:
                ep_rews.append(ep_rew)
                ep += 1
                if ep == 100:
                    print(
                        "Evaluating on {} episodes, Mean Reward: {} and Std Deviation for the reward: {}".format(
                            eval_ep, np.mean(ep_rews), np.std(ep_rews)
                        )
                    )
                    break

    def plot(self, results: List[float], window_size: int = 100) -> None:
        """
        plot model rewards
        :param results: rewards for each episode
        :param window_size: size of moving average filter
        :type results: int
        :type window_size: int
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
    # agent = QLearning(env, epsilon=0.6, lr=0.1)
    # trainer = Trainer(
    #     agent,
    #     env,
    #     mode="dyna",
    #     model="tabular",
    #     seed=42,
    #     n_episodes=1000,
    #     start_steps=0,
    #     evaluate_frequency=500,
    # )
    # ep_rs = trainer.train()
    # print("-" * 82)
    # print("Evaluating the learned model")
    # trainer.evaluate()
    # trainer.plot(ep_rs)
