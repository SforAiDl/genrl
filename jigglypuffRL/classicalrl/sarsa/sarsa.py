import gym
import numpy as np
import torch


class SARSA:
    """
    State-Action-Reward-State-Action (SARSA)
    :param env: (Gym environment) The environment to learn from
    :param max_iterations: (int) Maximum number of iterations per episode
    :param max_epsiodes: (int) Maximum number of episodes
    :param epsilon: (float) Epsilon for epsilon-greedy selection of the action
    :param alpha: (float) Learning rate
    :param gamma: (float) Discount Factor
    :param lmbda: (float) Lambda for the eligibility traces
    :param decay_rate_epsilon: (float) Decay rate for the epsilon of epsilon
        greedy action selection
    :param tensorboard_log: (str) the log location for tensorboard (if None,
        no logging)
    :param seed: (int) seed for torch and gym
    :param render: (bool) Need for rending of the environment while training
    """

    def __init__(
        self,
        env,
        max_iterations=1000,
        max_episodes=1500001,
        epsilon=0.9,
        alpha=0.1,
        gamma=0.95,
        lmbda=0.90,
        decay_rate_epsilon=0.999,
        tensorboard_log=None,
        seed=None,
        render=None,
    ):

        self.env = env
        self.max_iterations = max_iterations
        self.max_episodes = max_episodes
        self.epsilon = epsilon
        self.decay_rate_epsilon = decay_rate_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render

        # Assign seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            self.env.seed(seed)

        self.writer = None
        if self.tensorboard_log is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        # Set up the Q table
        self.Q_table = np.zeros((
            self.env.observation_space.n, self.env.action_space.n
        ))
        # Set up eligibility traces
        self.e_table = np.zeros((
            self.env.observation_space.n, self.env.action_space.n
        ))

    def select_action(self, state):
        # epsilon greedy method to sample actions
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state, :])

        return action

    def update_params(self, r, state1, action1, state2, action2):
        self.e_table[state1, action1] += 1
        delta = (
            r
            + self.gamma * self.Q_table[state2, action2]
            - self.Q_table[state1, action1]
        )
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                self.Q_table[s, a] = (
                    self.Q_table[s, a] + self.alpha * delta
                    * self.e_table[s, a]
                )
                self.e_table[s, a] = self.gamma*self.lmbda*self.e_table[s, a]

    def learn(self):
        for ep in range(self.max_episodes):
            t = 0

            state1 = self.env.reset()
            action1 = self.select_action(state1)

            for t in range(self.max_iterations):
                state2, reward, done, info = self.env.step(action1)

                if self.render:
                    self.env.render()

                action2 = self.select_action(state2)

                self.update_params(reward, state1, action1, state2, action2)

                state1 = state2
                action1 = action2

                if done:
                    self.epsilon = self.epsilon * self.decay_rate_epsilon
                    break

            if ep % 5000 == 0:
                # report every 5000 episodes, test 100 games to get average
                # point score for statistics
                rew_average = 0.0

                for i in range(100):
                    obs = self.env.reset()
                    done = False
                    while done is not True:
                        action = np.argmax(self.Q_table[obs])
                        obs, rew, done, info = self.env.step(action)
                        rew_average += rew
                rew_average = rew_average / 100
                print("Episode {} Reward: {}".format(ep, rew_average))

                if self.tensorboard_log:
                    self.writer.add_scalar("Reward", rew_average, ep)

            self.env.close()
        if self.tensorboard_log:
            self.writer.close()

        return self.Q_table


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    algo = SARSA(env)
    Q_table = algo.learn()
