import numpy as np
from scipy import stats
from typing import List, Dict, Any
from .contextual_bandits import ContextualBandit


class CBPolicy(object):
    """
    Base Class for Contextual Bandit solving Policy

    :param bandit: The Bandit to solve
    :param requires_init_run: Indicated if initialisation of Q values is required
    :type bandit: Bandit type object 
    """

    def __init__(self, bandit: ContextualBandit, requires_init_run: bool = False):
        self._bandit = bandit
        self._regret = 0.0
        self._action_hist = []
        self._regret_hist = []
        self._reward_hist = []

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
        self.action_hist.append(action)
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
        self.Q[context, action] += (reward - self.Q[context, action]) / (self.counts[context, action] + 1)
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
        super(UCBCBPolicy, self).__init__(bandit, requires_init_run=True)
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
            self.Q[context] + self.c * np.sqrt(2 * np.log(t + 1) / (self.counts[context] + 1))
        )
        self.action_hist.append(action)
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
        self.Q[context, action] += (reward - self.Q[context, action]) / (self.counts[context, action] + 1)
        self.counts[context, action] += 1

if __name__ == "__main__":

    def demo_policy(
        policy_type: CBPolicy,
        bandit_type: ContextualBandit,
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
            for i in range(iterations):
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
    from .contextual_bandits import BernoulliCB, GaussianCB

    timesteps = 10000
    iterations = 10
    bandits = 5
    arms = 10
    bandit_args = {"bandits": bandits, "arms": arms}

    eps_vals = [0.3] #[0.0, 0.01, 0.03, 0.1, 0.3]
    policy_args_collection = [{"eps": i} for i in eps_vals]
    demo_policy(
        EpsGreedyCBPolicy,
        BernoulliCB,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    c_vals = [0.5] #, 0.9, 1.0, 2.0]
    policy_args_collection = [{"c": i} for i in c_vals]
    demo_policy(
        UCBCBPolicy,
        BernoulliCB,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    eps_vals = [0.3] #[0.0, 0.01, 0.03, 0.1, 0.3]
    policy_args_collection = [{"eps": i} for i in eps_vals]
    demo_policy(
        EpsGreedyCBPolicy,
        GaussianCB,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )

    c_vals = [0.5] #, 0.9, 1.0, 2.0]
    policy_args_collection = [{"c": i} for i in c_vals]
    demo_policy(
        UCBCBPolicy,
        GaussianCB,
        policy_args_collection,
        bandit_args,
        timesteps,
        iterations,
    )
