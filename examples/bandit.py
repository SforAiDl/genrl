from genrl.classical.bandit import *
import numpy as np


def demo_policy(
    policy_type: BanditPolicy,
    bandit_type: Bandit,
    policy_args_collection,
    bandit_args,
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
