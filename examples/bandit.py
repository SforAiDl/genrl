from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from genrl.classical.bandit import *


def demo_policy(
    policy_type: Union[BanditPolicy, CBPolicy],
    bandit_type: Union[Bandit, ContextualBandit],
    policy_args_collection: Dict[str, Any],
    bandit_args: Dict[str, Any],
    timesteps: int,
    iterations: int,
    verbose: bool = False,
):
    """ Plots rewards and regrets of a given policy on given bandit """

    print(f"\nRunning {policy_type.__name__} on {bandit_type.__name__}")
    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    for policy_args in policy_args_collection:
        print(f"Running with policy parameters: = {policy_args}")
        average_reward = np.zeros(timesteps)
        average_regret = np.zeros(timesteps)
        for i in range(iterations):
            if verbose:
                print(f"Iteration {i}")
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
    plt.savefig(f"./logs/{policy_type.__name__}-on-{bandit_type.__name__}.png")
    plt.cla()


# Examples of regular bandits

TIMESTEPS = 1000
ITERATIONS = 500
ARMS = 10
BANDIT_ARGS = {"arms": ARMS}

eps_vals = [0.0, 0.01, 0.03, 0.1, 0.3]
POLICY_ARGS_COLLECTION = [{"eps": i} for i in eps_vals]
demo_policy(
    EpsGreedyPolicy,
    GaussianBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

c_vals = [0.5, 0.9, 1.0, 2.0]
POLICY_ARGS_COLLECTION = [{"c": i} for i in c_vals]
demo_policy(
    UCBPolicy,
    GaussianBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

alpha_vals = [0.1, 0.3]
temp_vals = [0.01, 0.1, 1.0]
POLICY_ARGS_COLLECTION = [
    {"alpha": i, "temp": j} for i in alpha_vals for j in temp_vals
]
demo_policy(
    GradientBasedPolicy,
    GaussianBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

eps_vals = [0.0, 0.01, 0.03, 0.1, 0.3]
POLICY_ARGS_COLLECTION = [{"eps": i} for i in eps_vals]
demo_policy(
    EpsGreedyPolicy,
    BernoulliBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

c_vals = [0.5, 0.9, 1.0, 2.0]
POLICY_ARGS_COLLECTION = [{"c": i} for i in c_vals]
demo_policy(
    UCBPolicy,
    GaussianBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

POLICY_ARGS_COLLECTION = [{"alpha": 1.0, "beta": 1.0, "c": 3.0}]
demo_policy(
    BayesianUCBPolicy,
    BernoulliBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

POLICY_ARGS_COLLECTION = [{"alpha": 1.0, "beta": 1.0}]
demo_policy(
    ThompsonSamplingPolicy,
    BernoulliBandit,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)


# Examples of contextual bandits

TIMESTEPS = 5000
ITERATIONS = 500
BANDITS = 5
ARMS = 10
BANDIT_ARGS = {"bandits": BANDITS, "arms": ARMS}

eps_vals = [0.3]  # [0.0, 0.01, 0.03, 0.1, 0.3]
POLICY_ARGS_COLLECTION = [{"eps": i} for i in eps_vals]
demo_policy(
    EpsGreedyCBPolicy,
    BernoulliCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

c_vals = [0.5]  # , 0.9, 1.0, 2.0]
POLICY_ARGS_COLLECTION = [{"c": i} for i in c_vals]
demo_policy(
    UCBCBPolicy,
    BernoulliCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

eps_vals = [0.3]  # [0.0, 0.01, 0.03, 0.1, 0.3]
POLICY_ARGS_COLLECTION = [{"eps": i} for i in eps_vals]
demo_policy(
    EpsGreedyCBPolicy,
    GaussianCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

c_vals = [0.5]  # , 0.9, 1.0, 2.0]
POLICY_ARGS_COLLECTION = [{"c": i} for i in c_vals]
demo_policy(
    UCBCBPolicy, GaussianCB, POLICY_ARGS_COLLECTION, BANDIT_ARGS, TIMESTEPS, ITERATIONS,
)

alpha_vals = [0.1, 0.3]
temp_vals = [0.01, 0.1, 1.0]
POLICY_ARGS_COLLECTION = [
    {"alpha": i, "temp": j} for i in alpha_vals for j in temp_vals
]
demo_policy(
    GradientBasedCBPolicy,
    GaussianCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

POLICY_ARGS_COLLECTION = [{"alpha": 1.0, "beta": 1.0, "c": 3.0}]
demo_policy(
    BayesianUCBCBPolicy,
    BernoulliCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)

POLICY_ARGS_COLLECTION = [{"alpha": 1.0, "beta": 1.0}]
demo_policy(
    ThompsonSamplingCBPolicy,
    BernoulliCB,
    POLICY_ARGS_COLLECTION,
    BANDIT_ARGS,
    TIMESTEPS,
    ITERATIONS,
)
