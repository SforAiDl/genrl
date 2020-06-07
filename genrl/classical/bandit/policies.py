import numpy as np
from scipy import stats
from typing import List, Dict, Any
from .bandits import Bandit


class BanditPolicy(object):
    """
    Base Class for Multi-armed Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of quality values is required
    :type bandit: Bandit type object
    """

    def __init__(self, bandit: Bandit, requires_init_run: bool = False):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []
        self._counts = np.zeros(self._bandit.arms)
        self._requires_init_run = requires_init_run

    @property
    def action_hist(self) -> List[int]:
        """
        Get the history of actions taken

        :returns: List of actions
        :rtype: list
        """
        return self._action_hist

    @property
    def regret_hist(self) -> List[float]:
        """
        Get the history of regrets computed for each step

        :returns: List of regrets
        :rtype: list
        """
        return self._regret_hist

    @property
    def regret(self) -> float:
        """
        Get the current regret

        :returns: The current regret
        :rtype: float
        """
        return self._regret

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received for each step

        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def counts(self) -> np.ndarray:
        """
        Get the number of times each action has been taken

        :returns: Numpy array with count for each action
        :rtype: numpy.ndarray
        """
        return self._counts

    def select_action(self, timestep: int) -> int:
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        raise NotImplementedError

    def learn(self, n_timesteps: int = 1000) -> None:
        """
        Learn to solve the environment over given number of timesteps

        Selects action, takes a step in the bandit and then updates
        the parameters according to the reward received. If policy
        requires an initial run, it takes each action once before starting

        :param n_timesteps: number of steps to learn for
        :type: int
        """
        if self._requires_init_run:
            for action in range(self._bandit.arms):
                reward = self._bandit.step(action)
                self.update_params(action, reward)
            n_timesteps -= self._bandit.arms

        for timestep in range(n_timesteps):
            action = self.select_action(timestep)
            reward = self._bandit.step(action)
            self.update_params(action, reward)


class EpsGreedyPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Epsilon Greedy Action Selection Strategy.

    Refer to Section 2.3 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param eps: Probability with which a random action is to be selected.
    :type bandit: Bandit type object
    :type eps: float
    """

    def __init__(self, bandit: Bandit, eps: float = 0.05):
        super(EpsGreedyPolicy, self).__init__(bandit)
        self._eps = eps
        self._quality = np.zeros(bandit.arms)

    @property
    def eps(self) -> float:
        """
        Get the asscoiated epsilon for the policy

        :returns: Probability with which a random action is to be selected
        :rtype: float
        """
        return self._eps

    @property
    def quality(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._quality

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to epsilon greedy startegy

        A random action is selected with espilon probability over
        the optimal action according to the current quality values to
        encourage exploration of the policy.

        :param t: timestep to choose action for
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.quality)
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
        that of the action. Updates the quality values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.quality[action] += (reward - self.quality[action]) / (self.counts[action] + 1)
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

    def __init__(self, bandit: Bandit, confidence: float = 1.0):
        super(UCBPolicy, self).__init__(bandit, requires_init_run=True)
        self._c = confidence
        self._quality = np.zeros(bandit.arms)

    @property
    def c(self) -> float:
        """
        Get the confidence level which weights the exploration term

        :returns: Confidence level which controls degree of exploration
        :rtype: float
        """
        return self._c

    @property
    def quality(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._quality

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to upper confidence bound action selction

        Take action that maximises a weighted sum of the quality values for the action
        and an exploration encouragement term controlled by c.

        :param timestep: timestep to choose action for
        :type timesteps: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.quality + self.c * np.sqrt(2 * np.log(timestep + 1) / (self.counts + 1))
        )
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
        that of the action. Updates the quality values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.quality[action] += (reward - self.quality[action]) / (self.counts[action] + 1)
        self.counts[action] += 1


class SoftmaxActionSelectionPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Softmax Action Selection Strategy.

    Refer to Section 2.8 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: The step size parameter for gradient based update
    :param temp: Temperature for softmax distribution over quality values of actions
    :type bandit: Bandit type object
    :type alpha: float
    :type temp: float
    """

    def __init__(self, bandit, alpha=0.1, temp=0.01):
        super(SoftmaxActionSelectionPolicy, self).__init__(
            bandit, requires_init_run=False
        )
        self._alpha = alpha
        self._temp = temp
        self._quality = np.zeros(bandit.arms)
        self._probability_hist = []

    @property
    def alpha(self) -> float:
        """
        Get the step size parameter for gradient based update of policy

        :returns: Step size which controls rate of learning for policy
        :rtype: float
        """
        return self._alpha

    @property
    def temp(self) -> float:
        """
        Get the temperature for softmax distribution over quality values of actions

        :returns: Temperature which controls softness of softmax distribution
        :rtype: float
        """
        return self._temp

    @property
    def quality(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._quality

    @property
    def probability_hist(self) -> np.ndarray:
        """
        Get the history of probabilty values assigned to each action for each timestep

        :returns: Numpy array of probability values for all actions
        :rtype: numpy.ndarray
        """
        return self._probability_hist

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        r"""
        Softmax with temperature
        :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i / temp)}{\sum_j \exp(x_j / temp)}`

        :param x: Set of values to compute softmax over
        :type x: numpy.ndarray
        :returns: Computed softmax over given values
        :rtype: numpy.ndarray
        """
        exp = np.exp(logits / self.temp)
        total = np.sum(exp)
        probabilities = exp / total
        return probabilities

    def select_action(self, timestep: int) -> int:
        """
        Select an action according by softmax action selection strategy

        Action is sampled from softmax distribution computed over
        the quality values for all actions

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        probabilities = self._softmax(self.quality)
        action = np.random.choice(self._bandit.arms, 1, p=probabilities)[0]
        self.action_hist.append(action)
        self.probability_hist.append(probabilities)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and that
        of the action. Updates the quality values through a gradient ascent step

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)

        # compute reward baseline by taking mean of all rewards till t-1
        if len(self.reward_hist) <= 1:
            reward_baseline = 0.0
        else:
            reward_baseline = np.mean(self.reward_hist[:-1])

        current_probailities = self.probability_hist[-1]

        # update quality values for the action taken and those not taken seperately
        self.quality[action] += (
            self.alpha * (reward - reward_baseline) * (1 - current_probailities[action])
        )
        actions_not_taken = np.arange(self._bandit.arms) != action
        self.quality[actions_not_taken] += (
            -1
            * self.alpha
            * (reward - reward_baseline)
            * current_probailities[actions_not_taken]
        )


class BayesianUCBPolicy(BanditPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: alpha value for beta distribution
    :param beta: beta values for beta distibution
    :param confidence: Confidence level which controls degree of exploration
    :type bandit: Bandit type object
    :type alpha: float
    :type beta: float
    :type confidence: float
    """

    def __init__(
        self,
        bandit: Bandit,
        alpha: float = 1.0,
        beta: float = 1.0,
        confidence: float = 3.0,
    ):
        super(BayesianUCBPolicy, self).__init__(bandit)
        self._c = confidence
        self._a = alpha * np.ones(self._bandit.arms)
        self._b = beta * np.ones(self._bandit.arms)

    @property
    def quality(self) -> np.ndarray:
        """
        Compute the q values for all the actions for alpha, beta and c

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self.a / (self.a + self.b)

    @property
    def a(self) -> np.ndarray:
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._a

    @property
    def b(self) -> np.ndarray:
        """
        Get the beta value of beta distribution associated with the policy

        :returns: beta values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._b

    @property
    def c(self) -> float:
        """
        Get the confidence level which weights the exploration term

        :returns: Confidence level which controls degree of exploration
        :rtype: float
        """
        return self._c

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to bayesian upper confidence bound

        Take action that maximises a weighted sum of the quality values and
        a beta distribution paramerterized by alpha and beta
        and weighted by c for each action.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(self.quality + stats.beta.std(self.a, self.b) * self.c)
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
        that of the action. Updates the quality values according to the
        reward recieved in this step.

        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self.a[action] += reward
        self.b[action] += 1 - reward
        self._regret += max(self.quality) - self.quality[action]
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

    def __init__(self, bandit: Bandit, alpha: float = 1.0, beta: float = 1.0):
        super(ThompsonSamplingPolicy, self).__init__(bandit)
        self._a = alpha * np.ones(self._bandit.arms)
        self._b = beta * np.ones(self._bandit.arms)

    @property
    def quality(self) -> np.ndarray:
        """
        Compute the q values for all the actions for alpha, beta and c

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self.a / (self.a + self.b)

    @property
    def a(self) -> np.ndarray:
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._a

    @property
    def b(self) -> np.ndarray:
        """
        Get the alpha value of beta distribution associated with the policy

        :returns: alpha values of the beta distribution
        :rtype: numpy.ndarray
        """
        return self._b

    def select_action(self, timestep: int) -> int:
        """
        Select an action according to Thompson Sampling

        Samples are taken from beta distribution parameterized by
        alpha and beta for each action. The action with the highest
        sample is selected.

        :param timestep: timestep to choose action for
        :type timestep: int
        :returns: Selected action
        :rtype: int
        """
        sample = np.random.beta(self.a, self.b)
        action = np.argmax(sample)
        self.action_hist.append(action)
        return action

    def update_params(self, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max quality value and
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
        self._regret += max(self.quality) - self.quality[action]
        self.regret_hist.append(self.regret)
        self.counts[action] += 1


if __name__ == "__main__":

    def demo_policy(
        policy_type: BanditPolicy,
        bandit_type: Bandit,
        policy_args_collection: Dict[str, Any],
        bandit_args: Dict[str, Any],
        timesteps: int,
        iterations: int,
    ):
        """ Plots rewards and regrets of a given policy on given bandit """

        print(f"\nRunning {policy_type.__name__} on {bandit_type.__name__}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for policy_args in policy_args_collection:
            print(f"Running with policy parameters: = {policy_args}")
            average_reward = np.zeros(timesteps)
            average_regret = np.zeros(timesteps)
            for _ in range(iterations):
                bandit = bandit_type(**bandit_args)
                policy = policy_type(bandit, **policy_args)
                policy.learn(timesteps)
                average_reward += np.array(policy.reward_hist) / iterations
                average_regret += np.array(policy.regret_hist) / iterations
            axs[0].plot(average_reward, label=f"{policy_args}")
            axs[1].plot(average_regret, label=f"{policy_args}")
        axs[0].legend()
        axs[1].legend()
        axs[0].set_title(f"{policy_type.__name__} Rewards on {bandit_type.__name__}")
        axs[1].set_title(f"{policy_type.__name__} Regrets on {bandit_type.__name__}")
        plt.savefig(f"{policy_type.__name__}-on-{bandit_type.__name__}.png")
        plt.cla()

    import matplotlib.pyplot as plt
    from .bandits import GaussianBandit, BernoulliBandit

    timesteps = 1000
    iterations = 2
    arms = 10
    bandit_args = {"arms": arms}

    eps_vals = [0.0, 0.01, 0.03, 0.1, 0.3]
    policy_args_collection = [{"eps": i} for i in eps_vals]
    demo_policy(
        EpsGreedyPolicy,
        GaussianBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    c_vals = [0.5, 0.9, 1.0, 2.0]
    policy_args_collection = [{"c": i} for i in c_vals]
    demo_policy(
        UCBPolicy,
        GaussianBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    alpha_vals = [0.1, 0.3]
    temp_vals = [0.01, 0.1, 1.0]
    policy_args_collection = [
        {"alpha": i, "temp": j} for i, j in zip(alpha_vals, temp_vals)
    ]
    demo_policy(
        SoftmaxActionSelectionPolicy,
        GaussianBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    eps_vals = [0.0, 0.01, 0.03, 0.1, 0.3]
    policy_args_collection = [{"eps": i} for i in eps_vals]
    demo_policy(
        EpsGreedyPolicy,
        BernoulliBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    c_vals = [0.5, 0.9, 1.0, 2.0]
    policy_args_collection = [{"c": i} for i in c_vals]
    demo_policy(
        UCBPolicy,
        GaussianBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    policy_args_collection = [{"alpha": 1.0, "beta": 1.0, "c": 3.0}]
    demo_policy(
        BayesianUCBPolicy,
        BernoulliBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    policy_args_collection = [{"alpha": 1.0, "beta": 1.0}]
    demo_policy(
        ThompsonSamplingPolicy,
        BernoulliBandit,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )
