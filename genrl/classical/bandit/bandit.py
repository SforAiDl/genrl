import numpy as np
from scipy.stats import beta


class Bandit(object):
    """
    Base Class for Multi-armed Bandits
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    """

    def __init__(self, bandits=1, arms=1):
        self._nbandits = bandits
        self._narms = arms
        self._regret = 0.0
        self._regrets = [0.0]
        self._avg_reward = []
        self._counts = np.zeros((bandits, arms))

    def learn(self, n_timesteps=None):
        raise NotImplementedError

    @property
    def arms(self):
        return self._narms

    @property
    def nbandits(self):
        return self._nbandits

    @property
    def regrets(self):
        return self._regrets

    @property
    def regret(self):
        return self._regret

    @property
    def avg_reward(self):
        return self._avg_reward

    @property
    def counts(self):
        return self._counts

    def get_action(self, t, bandit):
        raise NotImplementedError

    def get_reward(self, bandit, action):
        raise NotImplementedError

    def update(self, bandit, action, reward):
        raise NotImplementedError

    def step(self, t):
        bandit_rewards = []
        for bandit in range(self.nbandits):
            action = self.get_action(t, bandit)
            reward = self.get_reward(bandit, action)
            self.update(bandit, action, reward)
            bandit_rewards.append(reward)
        return bandit_rewards


class GaussianBandits(Bandit):
    """
    Multi-Armed Bandits with Stationary Rewards following a Gaussian
    distribution.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    """

    def __init__(self, bandits=1, arms=1):
        super(GaussianBandits, self).__init__(bandits, arms)
        self._rewards = np.random.normal(size=(bandits, arms))
        self._Q = np.zeros_like(self.rewards)
        self._counts = np.zeros_like(self.rewards)

    def learn(self, n_timesteps=None):
        raise NotImplementedError

    def update(self, bandit, action, reward):
        self._regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self.regrets.append(self.regret)
        self.Q[bandit, action] += (reward - self.Q[bandit, action]) / (
            self.counts[bandit, action] + 1
        )
        self.counts[bandit, action] += 1

    @property
    def Q(self):
        return self._Q

    @property
    def rewards(self):
        return self._rewards

    @property
    def avg_reward(self):
        return self._avg_reward

    def step(self, t):
        bandit_rewards = []
        for bandit in range(self.nbandits):
            action = self.get_action(t, bandit)
            reward = self.get_reward(bandit, action)
            self.update(bandit, action, reward)
            bandit_rewards.append(reward)
        return bandit_rewards


class EpsGreedyGaussianBandit(GaussianBandits):
    """
    Multi-Armed Bandit Solver with EpsGreedy Action Selection Strategy. Refer
    2.3 of Reinforcement Learning: An Introduction. Each arm is modeled as a
    Gaussian distribution.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    :param eps: (float) Probability with which a random action is to be
    selected.
    """

    def __init__(self, bandits=1, arms=10, eps=0.05):
        super(EpsGreedyGaussianBandit, self).__init__(bandits, arms)
        self._eps = eps

    def learn(self, n_timesteps=1000):
        for i in range(n_timesteps):
            r_step = self.step(i)
            self.avg_reward.append(np.mean(r_step))

    def get_action(self, t, bandit):
        if np.random.random() < self.eps:
            action = np.random.randint(0, self.arms)
        else:
            action = np.argmax(self.Q[bandit])
        return action

    def get_reward(self, bandit, action):
        reward = np.random.normal(self.rewards[bandit, action])
        return reward

    @property
    def eps(self):
        return self._eps


class UCBGaussianBandit(GaussianBandits):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound Based Action
    Selection. Refer 2.7 of Reinforcement Learning: An Introduction
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    """

    def __init__(self, bandits=1, arms=10):
        super(UCBGaussianBandit, self).__init__(bandits, arms)
        self._counts = np.zeros_like(self.rewards)

    def learn(self, n_timesteps=1000):
        self.initial_run()
        for i in range(n_timesteps):
            r = self.step(i)
            self.avg_reward.append(np.mean(r))

    def get_action(self, t, bandit):
        action = np.argmax(
            self.Q[bandit] + np.sqrt(2 * np.log(t + 1) / (self.counts[bandit] + 1))
        )
        return action

    def get_reward(self, bandit, action):
        reward = np.random.normal(self.rewards[bandit, action])
        return reward

    def initial_run(self):
        for bandit in range(self.nbandits):
            bandit_reward = []
            for arm in range(self.arms):
                reward = self.get_reward(bandit, arm)
                bandit_reward.append(reward)
                self.update(bandit, arm, reward)
            self.avg_reward.append(np.mean(bandit_reward))


class SoftmaxActionSelection(GaussianBandits):
    """
    Multi-Armed Bandit Softmax based Action Selection. Refer 2.8 of
    Reinforcement Learning: An Introduction
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    :param temp: (float) Temperature value for Softmax.
    """

    def __init__(self, bandits=1, arms=10, temp=0.01):
        super(SoftmaxActionSelection, self).__init__(bandits, arms)
        self._temp = temp

    def softmax(self, x):
        exp = np.exp(x / self.temp)
        total = np.sum(exp)
        return exp / total

    @property
    def temp(self):
        return self._temp

    def learn(self, n_timesteps=1000):
        self.initial_run()
        for i in range(n_timesteps):
            r = self.step(i)
            self.avg_reward.append(np.mean(r))

    def step(self, i):
        R_step = []
        for bandit in range(self.nbandits):
            action = self.get_action(i, bandit)
            reward = self.get_reward(bandit, action)
            R_step.append(reward)
            self.update(bandit, action, reward)
        return R_step

    def get_action(self, t, bandit):
        probabilities = self.softmax(self.Q[bandit])
        action = np.random.choice(range(self.arms), p=probabilities)
        return action

    def get_reward(self, bandit, action):
        reward = np.random.normal(self.rewards[bandit, action])
        return reward

    def initial_run(self):
        for bandit in range(self.nbandits):
            bandit_reward = []
            for arm in range(self.arms):
                reward = self.get_reward(bandit, arm)
                bandit_reward.append(reward)
                self.update(bandit, arm, reward)
            self.avg_reward.append(np.mean(bandit_reward))


class BernoulliBandits(Bandit):
    """
    Multi-Armed Bandits with Bernoulli probabilities.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    """

    def __init__(self, bandits=1, arms=10):
        super(BernoulliBandits, self).__init__(bandits, arms)
        self._init_probabilities = np.random.random()
        self._Q = self._init_probabilities * np.ones_like(self.counts)
        self._regrets = [0]
        self._regret = 0
        self._avg_reward = []

    def get_reward(self, bandit, arm):
        if np.random.random() < self.Q[bandit, arm]:
            return 1
        else:
            return 0

    def update(self, bandit, action, reward):
        self._regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self.regrets.append(self.regret)
        self.Q[bandit, action] += (
            1.0 / (self.counts[bandit, action] + 1) * (reward - self.Q[bandit, action])
        )
        self.counts[bandit, action] += 1

    @property
    def init_probabilities(self):
        return self._init_probabilities

    @property
    def Q(self):
        return self._Q

    def step(self, t):
        bandit_rewards = []
        for bandit in range(self.nbandits):
            action = self.get_action(t, bandit)
            reward = self.get_reward(bandit, action)
            self.update(bandit, action, reward)
            bandit_rewards.append(reward)
        return bandit_rewards


class EpsGreedyBernoulliBandit(BernoulliBandits):
    """
    Multi-Armed Bandit Solver with EpsGreedy Action Selection Strategy. Refer
    2.3 of Reinforcement Learning: An Introduction. Each arm is modeled as a
    Bernoulli RV.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    :param eps: (float) Probability with which a random action is to be
    selected.
    """

    def __init__(self, bandits=1, arms=10, eps=0.01):
        super(EpsGreedyBernoulliBandit, self).__init__(bandits, arms)
        self._eps = eps

    def learn(self, n_timesteps=1000):
        for t in range(n_timesteps):
            Rt = self.step(t)
            self.avg_reward.append(Rt)

    def get_action(self, t, bandit):
        if np.random.random() > self.eps:
            action = np.argmax(self.Q[bandit])
        else:
            action = np.random.randint(0, self.arms)
        return action

    @property
    def eps(self):
        return self._eps


class UCBBernoulliBandit(BernoulliBandits):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound Based Action
    Selection. Refer 2.7 of Reinforcement Learning: An Introduction. Each Arm
    modeled as a Bernoulli RV.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    """

    def __init__(self, bandits=1, arms=10):
        super(UCBBernoulliBandit, self).__init__(bandits, arms)

    def learn(self, n_timesteps=1000):
        self.initial_run()
        for t in range(n_timesteps):
            Rt = self.step(t)
            self.avg_reward.append(Rt)

    def get_action(self, t, bandit):
        action = np.argmax(
            self.Q[bandit] + np.sqrt(2 * np.log(t + 1) / (self.counts[bandit] + 1))
        )
        return action

    def initial_run(self):
        for bandit in range(self.nbandits):
            bandit_reward = []
            for arm in range(self.arms):
                reward = self.get_reward(bandit, arm)
                bandit_reward.append(reward)
                self.update(bandit, arm, reward)
            self.avg_reward.append(np.mean(bandit_reward))


class BayesianUCBBernoulliBandit(BernoulliBandits):
    """
    Multi-Armed Bandit Solver with Bayesian UCB action selection. Refer 2.7 of
    Reinforcement Learning: An Introduction. Each arm modeled as a Bernoulli
    RV.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    :param alpha: (int) alpha value of Beta distribution
    :param beta: (int) beta value of Beta distribution
    """

    def __init__(self, bandits, arms, a=1, b=1, c=3):
        super(BayesianUCBBernoulliBandit, self).__init__(bandits, arms)
        self._c = c
        self._a = a * np.ones_like(self.counts)
        self._b = b * np.ones_like(self.counts)

    @property
    def Q(self):
        return self.a / (self.a + self.b)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    def learn(self, n_timesteps=1000):
        for i in range(n_timesteps):
            r = self.step(i)
            self.avg_reward.append(np.mean(r))

    def get_action(self, t, bandit):
        return np.argmax(
            self.Q[bandit] + beta.std(self.a[bandit], self.b[bandit]) * self.c
        )

    def update(self, bandit, action, reward):
        self.a[bandit, action] += reward
        self.b[bandit, action] += 1 - reward
        self._regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self.regrets.append(self.regret)
        self.counts[bandit, action] += 1


class ThompsonSampling(BernoulliBandits):
    """
    Multi-Armed Bandit Solver with Thompson Sampling or Probability Matching.
    :param bandits: (int) Number of Bandits
    :param arms: (int) Number of arms in each bandit
    :param alpha: (int) alpha value of Beta distribution
    :param beta: (int) beta value of Beta distribution
    """

    def __init__(self, bandits=1, arms=10, alpha=1, beta=1):
        super(ThompsonSampling, self).__init__(bandits, arms)
        self._a = alpha * np.ones_like(self.counts)
        self._b = beta * np.ones_like(self.counts)

    @property
    def Q(self):
        return self.a / (self.a + self.b)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def learn(self, n_timesteps=1000):
        for i in range(n_timesteps):
            r = self.step(i)
            self.avg_reward.append(np.mean(r))

    def step(self, i):
        R_step = []
        for bandit in range(self.nbandits):
            sample = np.random.beta(self.a[bandit], self.b[bandit])
            action = self.get_action(sample)
            reward = self.get_reward(bandit, action)
            R_step.append(reward)
            self.update(bandit, action, reward)
        return R_step

    def get_action(self, sample):
        return np.argmax(sample)

    def update(self, bandit, action, reward):
        self.a[bandit, action] += reward
        self.b[bandit, action] += 1 - reward
        self._regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self.regrets.append(self.regret)
        self.counts[bandit, action] += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    epsGreedyBandit = EpsGreedyGaussianBandit(1, 10, 0.05)
    epsGreedyBandit.learn(1000)

    ucbBandit = UCBGaussianBandit(1, 100)
    ucbBandit.learn(1000)

    softmaxBandit = SoftmaxActionSelection(1, 10)
    softmaxBandit.learn(1000)

    plt.plot(epsGreedyBandit.regrets, label="eps greedy")
    plt.plot(ucbBandit.regrets, label="ucb")
    plt.plot(softmaxBandit.regrets, label="softmax")
    plt.legend()
    plt.savefig("GaussianBanditsRegret.png")
    plt.cla()

    epsbernoulli = EpsGreedyBernoulliBandit(1, 10, 0.05)
    epsbernoulli.learn(1000)

    ucbbernoulli = UCBBernoulliBandit(1, 10)
    ucbbernoulli.learn(1000)

    thsampling = ThompsonSampling(1, 10)
    thsampling.learn(1000)

    bayesianbandit = BayesianUCBBernoulliBandit(1, 10)
    bayesianbandit.learn(1000)

    plt.plot(epsbernoulli.regrets, label="eps")
    plt.plot(ucbbernoulli.regrets, label="ucb")
    plt.plot(bayesianbandit.regrets, label="Bayesian UCB")
    plt.plot(thsampling.regrets, label="Thompson Sampling")
    plt.legend()
    plt.savefig("BernoulliBanditsRegret.png")
