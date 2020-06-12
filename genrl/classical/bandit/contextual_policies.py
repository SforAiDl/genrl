from typing import List, Tuple

import numpy as np
from scipy import stats

from .contextual_bandits import ContextualBandit


class CBPolicy(object):
    """
    Base Class for Contextual Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of Q values is required
    :type bandit: Bandit type object
    """

    def __init__(self, bandit: ContextualBandit):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def action_hist(self) -> Tuple[int, int]:
        """
        Get the history of actions taken for contexts

        :returns: List of context, actions pairs
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

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action

        This method needs to be implemented in the specific policy.

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        raise NotImplementedError

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        This method needs to be implemented in the specific policy.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
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
        context = self._bandit.reset()
        for t in range(n_timesteps):
            action = self.select_action(context, t)
            context, reward = self._bandit.step(action)
            self.update_params(context, action, reward)


class EpsGreedyCBPolicy(CBPolicy):
    """
    Contextual Bandit Policy with Epsilon Greedy Action Selection Strategy.

    Refer to Section 2.3 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param eps: Probability with which a random action is to be selected.
    :type bandit: ContextualBandit type object
    :type eps: float
    """

    def __init__(self, bandit: ContextualBandit, eps: float = 0.05):
        super(EpsGreedyCBPolicy, self).__init__(bandit)
        self._eps = eps
        self._Q = np.zeros(shape=(bandit.bandits, bandit.arms))
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def eps(self) -> float:
        """
        Get the asscoiated epsilon for the policy

        :returns: Probability with which a random action is to be selected
        :rtype: float
        """
        return self._eps

    @property
    def Q(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according to epsilon greedy startegy

        A random action is selected with espilon probability over
        the optimal action according to the current Q values to
        encourage exploration of the policy.

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._bandit.arms)
        else:
            action = np.argmax(self.Q[context])
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.Q[context]) - self.Q[context, action]
        self.regret_hist.append(self.regret)
        self.Q[context, action] += (reward - self.Q[context, action]) / (
            self.counts[context, action] + 1
        )
        self.counts[context, action] += 1


class UCBCBPolicy(CBPolicy):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound based
    Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param c: Confidence level which controls degree of exploration
    :type bandit: ContextualBandit type object
    :type c: float
    """

    def __init__(self, bandit: ContextualBandit, c: float = 1.0):
        super(UCBCBPolicy, self).__init__(bandit)
        self._c = c
        self._Q = np.zeros(shape=(bandit.bandits, bandit.arms))
        self._counts = np.zeros(shape=(bandit.bandits, bandit.arms))

    @property
    def c(self) -> float:
        """
        Get the confidence level which weights the exploration term

        :returns: Confidence level which controls degree of exploration
        :rtype: float
        """
        return self._c

    @property
    def Q(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according to upper confidence bound action selction

        Take action that maximises a weighted sum of the Q values for the action
        and an exploration encouragement term controlled by c.

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.Q[context]
            + self.c * np.sqrt(2 * np.log(t + 1) / (self.counts[context] + 1))
        )
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward recieved in this step.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.Q[context]) - self.Q[context, action]
        self.regret_hist.append(self.regret)
        self.Q[context, action] += (reward - self.Q[context, action]) / (
            self.counts[context, action] + 1
        )
        self.counts[context, action] += 1


class GradientBasedCBPolicy(CBPolicy):
    """
    Multi-Armed Bandit Solver with Softmax Action Selection Strategy.

    Refer to Section 2.8 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: The step size parameter for gradient based update
    :param temp: Temperature for softmax distribution over Q values of actions
    :type bandit: ContextualBandit type object
    :type alpha: float
    :type temp: float
    """

    def __init__(self, bandit, alpha=0.1, temp=0.01):
        super(GradientBasedCBPolicy, self).__init__(bandit)
        self._alpha = alpha
        self._temp = temp
        self._Q = np.zeros(shape=(bandit.bandits, bandit.arms))
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
        Get the temperature for softmax distribution over Q values of actions

        :returns: Temperature which controls softness of softmax distribution
        :rtype: float
        """
        return self._temp

    @property
    def Q(self) -> np.ndarray:
        """
        Get the q values assigned by the policy to all actions

        :returns: Numpy array of q values for all actions
        :rtype: numpy.ndarray
        """
        return self._Q

    @property
    def probability_hist(self) -> np.ndarray:
        """
        Get the history of probabilty values assigned to each action for each timestep

        :returns: Numpy array of probability values for all actions
        :rtype: numpy.ndarray
        """
        return self._probability_hist

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        r"""
        Softmax with temperature
        :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i / temp)}{\sum_j \exp(x_j / temp)}`

        :param x: Set of values to compute softmax over
        :type x: numpy.ndarray
        :returns: Computed softmax over given values
        :rtype: numpy.ndarray
        """
        exp = np.exp(x / self.temp)
        total = np.sum(exp)
        p = exp / total
        return p

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according by softmax action selection strategy

        Action is sampled from softmax distribution computed over
        the Q values for all actions

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        probabilities = self._softmax(self.Q[context])
        action = np.random.choice(self._bandit.arms, 1, p=probabilities)[0]
        self.action_hist.append((context, action))
        self.probability_hist.append(probabilities)
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and that
        of the action. Updates the Q values through a gradient ascent step

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self._regret += max(self.Q[context]) - self.Q[context, action]
        self.regret_hist.append(self.regret)

        # compute reward baseline by taking mean of all rewards till t-1
        if len(self.reward_hist) <= 1:
            reward_baseline = 0.0
        else:
            reward_baseline = np.mean(self.reward_hist[:-1])

        current_probailities = self.probability_hist[-1]

        # update Q values for the action taken and those not taken seperately
        self.Q[context, action] += (
            self.alpha * (reward - reward_baseline) * (1 - current_probailities[action])
        )
        actions_not_taken = np.arange(self._bandit.arms) != action
        self.Q[context, actions_not_taken] += (
            -1
            * self.alpha
            * (reward - reward_baseline)
            * current_probailities[actions_not_taken]
        )


class BayesianUCBCBPolicy(CBPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    Refer to Section 2.7 of Reinforcement Learning: An Introduction.

    :param bandit: The Bandit to solve
    :param alpha: alpha value for beta distribution
    :param beta: beta values for beta distibution
    :param c: Confidence level which controls degree of exploration
    :type bandit: ContextualBandit type object
    :type alpha: float
    :type beta: float
    :type c: float
    """

    def __init__(
        self,
        bandit: ContextualBandit,
        alpha: float = 1.0,
        beta: float = 1.0,
        c: float = 3.0,
    ):
        super(BayesianUCBCBPolicy, self).__init__(bandit)
        self._c = c
        self._a = alpha * np.ones(shape=(bandit.bandits, bandit.arms))
        self._b = beta * np.ones(shape=(bandit.bandits, bandit.arms))

    @property
    def Q(self) -> np.ndarray:
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

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according to bayesian upper confidence bound

        Take action that maximises a weighted sum of the Q values and
        a beta distribution paramerterized by alpha and beta
        and weighted by c for each action

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        action = np.argmax(
            self.Q[context] + stats.beta.std(self.a[context], self.b[context]) * self.c
        )
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward recieved in this step

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self.a[context, action] += reward
        self.b[context, action] += 1 - reward
        self._regret += max(self.Q[context]) - self.Q[context, action]
        self.regret_hist.append(self.regret)
        self.counts[context, action] += 1


class ThompsonSamplingCBPolicy(CBPolicy):
    """
    Multi-Armed Bandit Solver with Bayesian Upper Confidence Bound
    based Action Selection Strategy.

    :param bandit: The Bandit to solve
    :param a: alpha value for beta distribution
    :param b: beta values for beta distibution
    :type bandit: ContextualBandit type object
    :type a: float
    :type b: float
    """

    def __init__(self, bandit: ContextualBandit, alpha: float = 1.0, beta: float = 1.0):
        super(ThompsonSamplingCBPolicy, self).__init__(bandit)
        self._a = alpha * np.ones(shape=(bandit.bandits, bandit.arms))
        self._b = beta * np.ones(shape=(bandit.bandits, bandit.arms))

    @property
    def Q(self) -> np.ndarray:
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

    def select_action(self, context: int, t: int) -> int:
        """
        Select an action according to Thompson Sampling

        Samples are taken from beta distribution parameterized by
        alpha and beta for each action. The action with the highest
        sample is selected.

        :param context: the context to select action for
        :param t: timestep to choose action for
        :type context: int
        :type t: int
        :returns: Selected action
        :rtype: int
        """
        sample = np.random.beta(self.a[context], self.b[context])
        action = np.argmax(sample)
        self.action_hist.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """
        Update parmeters for the policy

        Updates the regret as the difference between max Q value and
        that of the action. Updates the alpha value of beta distribution
        by adding the reward while the beta value is updated by adding
        1 - reward. Update the counts the action taken.

        :param context: context for which action is taken
        :param action: action taken for the step
        :param reward: reward obtained for the step
        :type context: int
        :type action: int
        :type reward: float
        """
        self.reward_hist.append(reward)
        self.a[context, action] += reward
        self.b[context, action] += 1 - reward
        self._regret += max(self.Q[context]) - self.Q[context, action]
        self.regret_hist.append(self.regret)
        self.counts[context, action] += 1
