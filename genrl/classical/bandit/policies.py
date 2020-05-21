import numpy as np
from scipy import stats


class BanditPolicy(object):
    """
    Base Class for Multi-armed Bandit solving Policy

    :param bandit: The Bandit to solve
    :type bandit: Bandit type object 
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
        """
        Get the history of actions taken

        :returns: List of actions
        :rtype: list
        """
        return self._action_hist

    @property
    def regret_hist(self):
        """
        Get the history of regrets computed for each step

        :returns: List of regrets
        :rtype: list
        """
        return self._regret_hist

    @property
    def regret(self):
        """
        Get the current regret

        :returns: The current regret
        :rtype: float
        """
        return self._regret

    @property
    def reward_hist(self):
        """
        Get the history of rewards received for each step

        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def counts(self):
        """
        Get the number of times each action has been taken

        :returns: Numpy array with count for each action
        :rtype: numpy.ndarray
        """
        return self._counts

    def select_action(self, t):
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        raise NotImplementedError

    def learn(self, n_timesteps=None):
        """
        Learn to solve the environment over given number of timesteps

        This method needs to be implemented in the specific policy.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        raise NotImplementedError


class EpsGreedyPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Epsilon Greedy Action Selection Strategy.

    Refer to Section 2.3 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param eps: Probability with which a random action is to be selected.
    :type bandit: Bandit type object 
    :type eps: float 
    """

    def __init__(self, bandit, eps=0.05):
        super(EpsGreedyPolicy, self).__init__(bandit)
        self._eps = eps
        self._Q = np.zeros(bandit.arms)

    @property
    def eps(self):
        """
        Get the asscoiated epsilon for the policy 

        :returns: Probability with which a random action is to be selected
        :rtype: float
        """
        return self._eps

    @property
    def Q(self):
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    def learn(self, n_timesteps=1000):
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)
            self.reward_hist.append(reward)

    def select_action(self, t):
        """
        Select an action according to epsilon greedy startegy

        A random action is selected with espilon probability over
        the optimal action according to the current Q values to
        encourage exploration of the policy.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.Q)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and 
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class UCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound based 
    Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param c: Confidence level which controls degree of exploration 
    :type bandit: Bandit type object 
    :type c: float 
    """

    def __init__(self, bandit, c=1):
        super(UCBPolicy, self).__init__(bandit)
        self._c = c
        self._Q = np.zeros(bandit.arms)

    @property
    def c(self):
        """
        Get the confidence level which weights the exploration term 
        
        :returns: Confidence level which controls degree of exploration 
        :rtype: float
        """
        return self._c

    @property
    def Q(self):
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    def learn(self, n_timesteps=1000):
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        self._initial_run()
        for t in range(n_timesteps - self._bandit.arms):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def select_action(self, t):
        """
        Select an action according to upper confidence bound action selction

        Take action that maximises a weighted sum of the Q values for the action
        and an exploration encouragement term controlled by c.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.Q + self.c * np.sqrt(2 * np.log(t + 1) / (self.counts + 1))
        )
        self.action_hist.append(action)
        return action

    def _initial_run(self):
        """
        Takes each action once to initialise Q valuess
        """
        for action in range(self._bandit.arms):
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and 
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class SoftmaxActionSelectionPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Softmax Action Selection Strategy.

    Refer to Section 2.8 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param temp: Temperature for softmax distribution over Q values of actions
    :type bandit: Bandit type object 
    :type temp: float 
    """

    def __init__(self, bandit, temp=0.01):
        super(SoftmaxActionSelectionPolicy, self).__init__(bandit)
        self._temp = temp
        self._Q = np.zeros(bandit.arms)

    @property
    def temp(self):
        """
        Get the temperature for softmax distribution over Q values of actions
        
        :returns: Temperature which controls softness of softmax distribution
        :rtype: float
        """
        return self._temp

    @property
    def Q(self):
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    def _softmax(self, x):
        r"""
        Softmax with temperature
        :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i / temp)}{\sum_j \exp(x_j / temp)}`

        :param x: Set of values to compute softmax over
        :type x: numpy.ndarray
        """
        exp = np.exp(x / self.temp)
        total = np.sum(exp)
        return exp / total

    def learn(self, n_timesteps=1000):
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        self._initial_run()
        for t in range(n_timesteps - self._bandit.arms):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def select_action(self, t):
        """
        Select an action according by softmax action selection strategy

        Action is sampled from softmax distribution computed over 
        the Q values for all actions

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        probabilities = self._softmax(self.Q)
        action = np.random.choice(range(self._bandit.arms), p=probabilities)
        self.action_hist.append(action)
        return action

    def _initial_run(self):
        """
        Takes each action once to initialise Q valuess
        """
        for action in range(self._bandit.arms):
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and 
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.Q[action] += (reward - self.Q[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class BayesianUCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound 
    based Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param a: alpha value for beta distribution
    :param b: beta values for beta distibution
    :param c: Confidence level which controls degree of exploration 
    :type bandit: Bandit type object 
    :type a: float
    :type b: float
    :type c: float 
    """

    def __init__(self, bandit, a=1, b=1, c=3):
        super(BayesianUCBPolicy, self).__init__(bandit)
        self._c = c
        self._a = a * np.ones(self._bandit.arms)
        self._b = b * np.ones(self._bandit.arms)

    @property
    def Q(self):
        """
        Compute the q values for all the actions for alpha, beta and c

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self.a / (self.a + self.b)

    @property
    def a(self):
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: np.ndarray 
        """
        return self._a

    @property
    def b(self):
        """
        Get the beta value of beta distribution associated with the policy

        :returns: beta values of the beta distribution
        :rtype: np.ndarray 
        """
        return self._b

    @property
    def c(self):
        """
        Get the confidence level which weights the exploration term 
        
        :returns: Confidence level which controls degree of exploration 
        :rtype: float
        """
        return self._c

    def learn(self, n_timesteps=1000):
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def select_action(self, t):
        """
        Select an action according to bayesian upper confidence bound

        Take action that maximises a weighted sum of the Q values and 
        a beta distribution paramerterized by alpha and beta 
        and weighted by c for each action.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(self.Q + stats.beta.std(self.a, self.b) * self.c)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and 
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.Q) - self.Q[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1


class ThompsonSamplingPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound 
    based Action Selection Strategy.

    :param bandit: The Bandit to solve
    :param a: alpha value for beta distribution
    :param b: beta values for beta distibution
    :type bandit: Bandit type object 
    :type a: float
    :type b: float
    """

    def __init__(self, bandit, alpha=1, beta=1):
        super(ThompsonSamplingPolicy, self).__init__(bandit)
        self._a = alpha * np.ones(self._bandit.arms)
        self._b = beta * np.ones(self._bandit.arms)

    @property
    def Q(self):
        """
        Compute the q values for all the actions for alpha, beta and c

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self.a / (self.a + self.b)

    @property
    def a(self):
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: np.ndarray 
        """
        return self._a

    @property
    def b(self):
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: np.ndarray 
        """
        return self._b

    def learn(self, n_timesteps=1000):
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received.

        :param n_timesteps: number of steps to learn for
        :type: int 
        """
        for t in range(n_timesteps):
            action = self.select_action(t)
            reward = self._bandit.step(action)
            self.update_params(action, reward)

    def select_action(self, t):
        """
        Select an action according to Thompson Sampling

        Samples are taken from beta distribution parameterized by
        alpha and beta for each action. The action with the highest
        sample is selected.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        sample = np.random.beta(self.a, self.b)
        action = np.argmax(sample)
        self.action_hist.append(action)
        return action

    def update_params(self, action, reward):
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and 
        that of the action. Updates the alpha value of beta distribution
        by adding the reward while the beta value is updated by adding
        1 - reward. Update the counts the action taken.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: int
        """
        self.reward_hist.append(reward)
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
    show = False

    print(f"\nRunning Epsillon Greedy Policy on Gaussian Bandit")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
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
        axs[0].plot(average_reward, label=f"{eps}")
        axs[1].plot(average_regret, label=f"{eps}")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Eps Greedy Rewards on Gaussian Bandit")
    axs[1].set_title("Eps Greedy Regrets on Gaussian Bandit")
    plt.savefig("GaussianEpsGreedyPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning UCB Policy on Gaussian Bandit")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for c in [0.5, 0.9, 1, 2]:
        print(f"Running for c = {c}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            gaussian_bandit = GaussianBandit(arms)
            ucb_gaussian = UCBPolicy(gaussian_bandit, c)
            ucb_gaussian.learn(timesteps)
            average_reward += np.array(ucb_gaussian.reward_hist) / iterations
            average_regret += np.array(ucb_gaussian.regret_hist) / iterations
        axs[0].plot(average_reward, label=f"{c}")
        axs[1].plot(average_regret, label=f"{c}")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("UCB Rewards on Gaussian Bandit")
    axs[1].set_title("UCB Regrets on Gaussian Bandit")
    plt.savefig("GaussianUCBPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning Softmax Selection Policy on Gaussian Bandit")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for temp in [0.01, 0.03, 0.1]:
        print(f"Running for temp = {temp}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            gaussian_bandit = GaussianBandit(arms)
            softmax_gaussian = SoftmaxActionSelectionPolicy(gaussian_bandit, temp)
            softmax_gaussian.learn(timesteps)
            average_reward += np.array(softmax_gaussian.reward_hist) / iterations
            average_regret += np.array(softmax_gaussian.regret_hist) / iterations
        axs[0].plot(average_reward, label=f"{temp}")
        axs[1].plot(average_regret, label=f"{temp}")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Softmax Selection Rewards on Gaussian Bandit")
    axs[1].set_title("Softmax Selection Regrets on Gaussian Bandit")
    plt.savefig("GaussianSoftmaxPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning Epsillon Greedy Policy on Bernoulli Bandit")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
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
        axs[0].plot(average_reward, label=f"{eps}")
        axs[1].plot(average_regret, label=f"{eps}")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Eps Greedy Rewards on Bernoulli Bandit")
    axs[1].set_title("Eps Greedy Regrets on Bernoulli Bandit")
    plt.savefig("BernoulliEpsGreedyPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning UCB Policy on Bernoulli Bandit")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for c in [0.5, 0.9, 1, 2]:
        print(f"Running for c = {c}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            bernoulli_bandit = BernoulliBandit(arms)
            ucb_bernoulli = UCBPolicy(bernoulli_bandit, c)
            ucb_bernoulli.learn(timesteps)
            average_reward += np.array(ucb_bernoulli.reward_hist) / iterations
            average_regret += np.array(ucb_bernoulli.regret_hist) / iterations
        axs[0].plot(average_reward, label=f"{c}")
        axs[1].plot(average_regret, label=f"{c}")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("UCB Rewards on Bernoulli Bandit")
    axs[1].set_title("UCB Regrets on Bernoulli Bandit")
    plt.savefig("BernoulliUCBPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning Bayesian UCB Policy on Bernoulli Bandit")
    average_reward = np.zeros(timesteps)
    average_regret = np.zeros(timesteps)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(iterations):
        bernoulli_bandit = BernoulliBandit(arms)
        bayesian_ucb_bernoulli = BayesianUCBPolicy(bernoulli_bandit)
        bayesian_ucb_bernoulli.learn(timesteps)
        average_reward += np.array(bayesian_ucb_bernoulli.reward_hist) / iterations
        average_regret += np.array(bayesian_ucb_bernoulli.regret_hist) / iterations
    axs[0].plot(average_reward)
    axs[0].set_title("Bayesian UCB Rewards on Bernoulli Bandit")
    axs[1].plot(average_regret)
    axs[1].set_title("Bayesian UCB Regrets on Bernoulli Bandit")
    plt.savefig("BernoulliBayesianUCBPolicy.png")
    if show:
        plt.show()
    plt.cla()

    print(f"\nRunning Thompson Sampling Policy on Bernoulli Bandit")
    average_reward = np.zeros(timesteps)
    average_regret = np.zeros(timesteps)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(iterations):
        bernoulli_bandit = BernoulliBandit(arms)
        thompson_sampling_bernoulli = ThompsonSamplingPolicy(bernoulli_bandit)
        thompson_sampling_bernoulli.learn(timesteps)
        average_reward += np.array(thompson_sampling_bernoulli.reward_hist) / iterations
        average_regret += np.array(thompson_sampling_bernoulli.regret_hist) / iterations
    print(average_regret)
    axs[0].plot(average_reward)
    axs[0].set_title("Thompson Sampling Rewards on Bernoulli Bandit")
    axs[1].plot(average_regret)
    axs[1].set_title("Thompson Sampling Regrets on Bernoulli Bandit")
    plt.savefig("BernoulliThompsonSamplingPolicy.png")
    if show:
        plt.show()
    plt.cla()
