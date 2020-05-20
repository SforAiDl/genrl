import numpy as np
from scipy import stats


class BanditPolicy(object):
    """
    Base Class for Multi-armed Bandit solving Policy
    :param bandit: (Bandit) The Bandit to solve
    """

    def __init__(self, bandit):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []
        self._counts = np.zeros(self._bandit.arms)

    @property
    def action_hist(self):
        return self._action_hist

    @property
    def regret_hist(self):
        return self._regret_hist

    @property
    def regret(self):
        return self._regret

    @property
    def reward_hist(self):
        return self._reward_hist

    @property
    def counts(self):
        return self._counts

    def select_action(self, t):
        raise NotImplementedError

    def update_params(self, action, reward):
        raise NotImplementedError

    def learn(self, n_timesteps=None):
        raise NotImplementedError


class EpsGreedyPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with EpsGreedy Action Selection Strategy. Refer
    2.3 of Reinforcement Learning: An Introduction.
    :param bandit: (Bandit) The Bandit to solve
    :param eps: (float) Probability with which a random action is to be
    selected.
    """

    def __init__(self, bandit, eps=0.05):
        super(EpsGreedyPolicy, self).__init__(bandit)
        self._eps = eps
        self._Q = np.zeros(bandit.arms)

    @property
    def eps(self):
        return self._eps

    @property
    def Q(self):
        return self._Q

    def learn(self, n_timesteps=1000):
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.Q)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class UCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound Based Action
    Selection. Refer 2.7 of Reinforcement Learning: An Introduction
    :param bandit: (Bandit) The Bandit to solve
    """

    def __init__(self, bandit, c=1):
        super(UCBPolicy, self).__init__(bandit)
        self._c = c
        self._Q = np.zeros(bandit.arms)

    @property
    def c(self):
        return self._c

    @property
    def Q(self):
        return self._Q

    def learn(self, n_timesteps=1000):
        self._initial_run()
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        action = np.argmax(
            self.Q + self.c * np.sqrt(2 * np.log(t + 1) / (self.counts + 1))
        )
        self.action_hist.append(action)
        return action

    def _initial_run(self):
        for action in range(self._bandit.arms):
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def update_params(self, action, reward):
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class SoftmaxActionSelectionPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Softmax based Action Selection. Refer 2.8 of
    Reinforcement Learning: An Introduction
    :param bandit: (Bandit) The Bandit to solve
    :param temp: (float) Temperature value for Softmax.
    """

    def __init__(self, bandit, temp=0.01):
        super(SoftmaxActionSelectionPolicy, self).__init__(bandit)
        self._temp = temp
        self._Q = np.zeros(bandit.arms)

    @property
    def temp(self):
        return self._temp

    @property
    def Q(self):
        return self._Q

    def _softmax(self, x):
        exp = np.exp(x / self.temp)
        total = np.sum(exp)
        return exp / total

    def learn(self, n_timesteps=1000):
        self._initial_run()
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        probabilities = self._softmax(self.Q)
        action = np.random.choice(range(self._bandit.arms), p=probabilities)
        self.action_hist.append(action)
        return action

    def _initial_run(self):
        for action in range(self._bandit.arms):
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def update_params(self, action, reward):
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class BayesianUCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian UCB action selection. Refer 2.7 of
    Reinforcement Learning: An Introduction.
    :param bandit: (Bandit) The Bandit to solve
    :param arms: (int) Number of arms in each bandit
    :param alpha: (int) alpha value of Beta distribution
    :param beta: (int) beta value of Beta distribution
    """

    def __init__(self, bandit, a=1, b=1, c=3):
        super(BayesianUCBPolicy, self).__init__(bandit)
        self._c = c
        self._a = a * np.ones(self._bandit.arms)
        self._b = b * np.ones(self._bandit.arms)

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
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        action = np.argmax(self.Q + stats.beta.std(self.a, self.b) * self.c)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1


class ThompsonSamplingPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Thompson Sampling or Probability Matching.
    :param bandit: (Bandit) The Bandit to solve
    :param alpha: (int) alpha value of Beta distribution
    :param beta: (int) beta value of Beta distribution
    """

    def __init__(self, bandit, alpha=1, beta=1):
        super(ThompsonSamplingPolicy, self).__init__(bandit)
        self._a = alpha * np.ones(self._bandit.arms)
        self._b = beta * np.ones(self._bandit.arms)

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
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        sample = np.random.beta(self.a, self.b)
        action = np.argmax(sample)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bandits import GaussianBandit, BernoulliBandit

    timesteps = 1000
    iterations = 2000
    arms = 10

    print(f"\nRunning Epsillon Greedy Policy on Gaussian Bandit")
    for eps in [0, 0.01, 0.03, 0.1, 0.3]:
        print(f"Running for eps = {eps}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            gaussian_bandit = GaussianBandit(arms)
            eps_greedy_gaussian = EpsGreedyPolicy(gaussian_bandit, eps)
            eps_greedy_gaussian.learn(timesteps)
            average_reward += np.array(eps_greedy_gaussian.reward_hist) / iterations
            average_regret += np.array(eps_greedy_gaussian.regret_hist) / iterations
        plt.plot(average_reward, label=f"{eps}")
    plt.legend()
    plt.title("Eps Greedy Rewards on Gaussian Bandit")
    plt.savefig("GaussianEpsGreedyPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning UCB Policy on Gaussian Bandit")
    for c in [0.5, 0.9, 1, 2]:
        print(f"Running for c = {c}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            gaussian_bandit = GaussianBandit(arms)
            ucb_gaussian = UCBPolicy(gaussian_bandit, c)
            ucb_gaussian.learn(timesteps)
            average_reward += np.array(ucb_gaussian.reward_hist) / iterations
            # average_regret += np.array(ucb_gaussian.regret_hist) / iterations
        plt.plot(average_reward, label=f"{c}")
    plt.legend()
    plt.title("UCB Rewards on Gaussian Bandit")
    plt.savefig("GaussianUCBPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning Softmax Selection Policy on Gaussian Bandit")
    for temp in [0.01, 0.03, 0.1]:
        print(f"Running for temp = {temp}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            gaussian_bandit = GaussianBandit(arms)
            softmax_gaussian = SoftmaxActionSelectionPolicy(gaussian_bandit, temp)
            softmax_gaussian.learn(timesteps)
            average_reward += np.array(softmax_gaussian.reward_hist) / iterations
            # average_regret += np.array(softmax_gaussian.regret_hist) / iterations
        plt.plot(average_reward, label=f"{temp}")
    plt.legend()
    plt.title("Softmax Selection Rewards on Gaussian Bandit")
    plt.savefig("GaussianSoftmaxPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning Epsillon Greedy Policy on Bernoulli Bandit")
    for eps in [0, 0.01, 0.03, 0.1, 0.3]:
        print(f"Running for eps = {eps}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            bernoulli_bandit = BernoulliBandit(arms)
            eps_greedy_bernoulli = EpsGreedyPolicy(bernoulli_bandit, eps)
            eps_greedy_bernoulli.learn(timesteps)
            average_reward += np.array(eps_greedy_bernoulli.reward_hist) / iterations
            average_regret += np.array(eps_greedy_bernoulli.regret_hist) / iterations
        plt.plot(average_reward, label=f"{eps}")
    plt.legend()
    plt.title("Eps Greedy Rewards on Bernoulli Bandit")
    plt.savefig("BernoulliEpsGreedyPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning UCB Policy on Bernoulli Bandit")
    for c in [0.5, 0.9, 1, 2]:
        print(f"Running for c = {c}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            bernoulli_bandit = BernoulliBandit(arms)
            ucb_bernoulli = UCBPolicy(bernoulli_bandit, c)
            ucb_bernoulli.learn(timesteps)
            average_reward += np.array(ucb_bernoulli.reward_hist) / iterations
            # average_regret += np.array(ucb_bernoulli.regret_hist) / iterations
        plt.plot(average_reward, label=f"{c}")
    plt.legend()
    plt.title("UCB Rewards on Bernoulli Bandit")
    plt.savefig("BernoulliUCBPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning Bayesian UCB Policy on Bernoulli Bandit")
    average_reward = np.zeros(timesteps)
    average_regret = np.zeros(timesteps)
    for i in range(iterations):
        bernoulli_bandit = BernoulliBandit(arms)
        bayesian_ucb_bernoulli = BayesianUCBPolicy(bernoulli_bandit)
        bayesian_ucb_bernoulli.learn(timesteps)
        average_reward += np.array(bayesian_ucb_bernoulli.reward_hist) / iterations
        # average_regret += np.array(bayesian_ucb_bernoulli.regret_hist) / iterations
    plt.plot(average_reward)
    plt.title("Bayesian UCB Rewards on Bernoulli Bandit")
    plt.savefig("BernoulliBayesianUCBPolicyRewards.png")
    # plt.show()
    plt.cla()

    print(f"\nRunning Thompson Sampling Policy on Bernoulli Bandit")
    average_reward = np.zeros(timesteps)
    average_regret = np.zeros(timesteps)
    for i in range(iterations):
        bernoulli_bandit = BernoulliBandit(arms)
        thompson_sampling_bernoulli = ThompsonSamplingPolicy(bernoulli_bandit)
        thompson_sampling_bernoulli.learn(timesteps)
        average_reward += np.array(thompson_sampling_bernoulli.reward_hist) / iterations
        # average_regret += np.array(thompson_sampling_bernoulli.regret_hist) / iterations
    plt.plot(average_reward)
    plt.title("Thompson Sampling Rewards on Bernoulli Bandit")
    plt.savefig("BernoulliThompsonSamplingPolicyRewards.png")
    # plt.show()
    plt.cla()
