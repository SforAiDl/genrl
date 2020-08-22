from pathlib import Path

from matplotlib import pyplot as plt

import genrl

bandit = genrl.bandit.CovertypeDataBandit()
agent = genrl.bandit.NeuralLinearPosteriorAgent(bandit)
trainer = genrl.bandit.DCBTrainer(
    agent, bandit, logdir="logs/", log_mode=["stdout", "tensorboard"]
)
results = trainer.train(timesteps=100)

fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
fig.suptitle("Deep Contextual Bandit Example", fontsize=14)
axs[0, 0].set_title("Cumulative Regret")
axs[0, 1].set_title("Cumulative Reward")
axs[1, 0].set_title("Regret Moving Avg")
axs[1, 1].set_title("Reward Moving Avg")
axs[0, 0].plot(results["cumulative_regrets"], label="neural-linpos")
axs[0, 1].plot(results["cumulative_rewards"], label="neural-linpos")
axs[1, 0].plot(results["regret_moving_avgs"], label="neural-linpos")
axs[1, 1].plot(results["reward_moving_avgs"], label="neural-linpos")

plt.legend()
fig.savefig(Path("logs/").joinpath("dcb_example.png"))
