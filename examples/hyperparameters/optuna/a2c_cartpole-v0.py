import numpy as np
import optuna
import torch

from genrl.agents.a2c.a2c import A2C
from genrl.environments.suite import VectorEnv
from genrl.trainers.onpolicy import OnPolicyTrainer

env = VectorEnv("CartPole-v0")


def tune_A2C(trial):
    # Define hyperparameters that are relevant for training
    # Choose a suggestion type and range (float/int and log/uniform)
    lr_value = trial.suggest_float("lr_value", 1e-5, 1e-2, log=True)
    lr_policy = trial.suggest_float("lr_policy", 1e-5, 1e-2, log=True)
    rollout_size = trial.suggest_int("rollout_size", 100, 10000, log=True)
    entropy_coeff = trial.suggest_float("entropy_coeff", 5e-4, 2e-1, log=True)

    agent = A2C(
        "mlp",
        env,
        lr_value=lr_value,
        lr_policy=lr_policy,
        rollout_size=rollout_size,
        entropy_coeff=entropy_coeff,
    )
    trainer = OnPolicyTrainer(
        agent,
        env,
        log_interval=10,
        epochs=100,
        evaluate_episodes=10,
    )
    trainer.train()

    episode, episode_reward = 0, np.zeros(trainer.env.n_envs)
    episode_rewards = []
    state = trainer.env.reset()
    while True:
        if trainer.off_policy:
            action = trainer.agent.select_action(state, deterministic=True)
        else:
            action, _, _ = trainer.agent.select_action(state)

        if isinstance(action, torch.Tensor):
            action = action.numpy()

        next_state, reward, done, _ = trainer.env.step(action)

        episode_reward += reward
        state = next_state
        if np.any(done):
            for i, di in enumerate(done):
                if di:
                    episode += 1
                    episode_rewards.append(episode_reward[i])
                    episode_reward[i] = 0
        if episode == trainer.evaluate_episodes:
            print(
                "Evaluated for {} episodes, Mean Reward: {:.2f}, Std Deviation for the Reward: {:.2f}".format(
                    trainer.evaluate_episodes,
                    np.mean(episode_rewards),
                    np.std(episode_rewards),
                )
            )
            break

    return np.mean(episode_rewards)


agent_name = "A2C"  # replace
study_name = "{}-3".format(agent_name)
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    storage="sqlite:///{}.db".format(study_name),
    # load_if_exists=True
)
study.optimize(tune_A2C, n_trials=20)

print("Best Trial Results:")
for key, value in study.best_trial.__dict__.items():
    print("{} : {}".format(key, value))
