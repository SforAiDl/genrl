import numpy as np


class Bandit(object):
    def __init__(self, bandits=1, arms=1):
        self._nbandits = bandits
        self._narms = arms

    def learn(self, n_timesteps=None):
        raise NotImplementedError

    @property
    def arms(self):
        return self._narms

    @property
    def nbandits(self):
        return self._nbandits


class GaussianBandits(Bandit):
    def __init__(self, bandits=1, arms=1):
        super(GaussianBandits, self).__init__(bandits, arms)
        self._rewards = np.random.normal(size=(bandits, arms))
        self._Q = np.zeros_like(self.rewards)
        self._counts = np.zeros_like(self.rewards)

    def learn(self, n_timesteps=None):
        raise NotImplementedError

    @property
    def Q(self):
        return self._Q

    @property
    def rewards(self):
        return self._rewards

    @property
    def counts(self):
        return self._counts


class EpsGreedy(GaussianBandits):
    def __init__(self, bandits=1, arms=10, eps=0.05):
        super(EpsGreedy, self).__init__(bandits, arms)
        self._eps = eps
        self.regret = 0.0
        self.regrets = [0.0]

    def learn(self, n_timesteps=1000):
        self.avg_reward = []
        for i in range(n_timesteps):
            r_step = self.one_step(i)
            self.avg_reward.append(np.mean(r_step))

    def one_step(self, i):
        R_step = []
        for bandit in range(self.nbandits):
            action = self.get_action(bandit)
            reward = self.get_reward(bandit, action)
            R_step.append(reward)
            self.update_regret(bandit, action)
            self.Q[bandit, action] += (reward - self.Q[bandit, action]) / (
                self.counts[bandit, action] + 1
            )
            self.counts[bandit, action] += 1
        return R_step

    def get_action(self, bandit):
        if np.random.random() < self.eps:
            action = np.random.randint(0, self.arms)
        else:
            action = np.argmax(self.Q[bandit])
        return action

    def get_reward(self, bandit, action):
        reward = np.random.normal(self.rewards[bandit, action])
        return reward

    def update_regret(self, bandit, action):
        self.regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self.regrets.append(self.regret)

    @property
    def eps(self):
        return self._eps


class UCB(GaussianBandits):
    def __init__(self, bandits=1, arms=10):
        super(UCB, self).__init__(bandits, arms)
        self._counts = np.zeros_like(self.rewards)
        self.regret = 0.0
        self._regrets = [0.0]

    def learn(self, n_timesteps=1000):
        self.avg_reward = []
        for i in range(n_timesteps):
            r = self.one_step(i)
            self.avg_reward.append(np.mean(r))

    def one_step(self, i):
        R_step = []
        for bandit in range(self.nbandits):
            action = self.get_action(i, bandit)
            self.counts[bandit, action] += 1
            reward = self.get_reward(bandit, action)
            R_step.append(reward)
            self.Q[bandit, action] += (reward - self.Q[bandit, action]) / (
                self.counts[bandit, action] + 1
            )
            self.counts[bandit, action] += 1
            self.update_regret(bandit, action)
        return R_step

    def get_action(self, t, bandit):
        action = np.argmax(
            self.Q[bandit] + np.sqrt(2 * np.log(t) / self.counts[bandit])
        )
        return action

    def get_reward(self, bandit, action):
        reward = np.random.normal(self.rewards[bandit, action])
        return reward

    def update_regret(self, bandit, action):
        self.regret += max(self.Q[bandit]) - self.Q[bandit][action]
        self._regrets.append(self.regret)

    @property
    def counts(self):
        return self._counts

    @property
    def regrets(self):
        return self._regrets
