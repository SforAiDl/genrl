import numpy as np
import optuna
import torch

from genrl.agents.td3.td3 import TD3
from genrl.environments.suite import VectorEnv
from genrl.trainers.offpolicy import OffPolicyTrainer

env = VectorEnv("Pendulum-v0")


def objective(trial):
    lr_value = trial.suggest_float("lr_value", 1e-6, 1e-1, log=True)
    lr_policy = trial.suggest_float("lr_policy", 1e-6, 1e-1, log=True)
    replay_size = trial.suggest_int("replay_size", 1e2, 1e5, log=True)
    max_ep_len = trial.suggest_int("max_ep_len", 1e3, 50000, log=True)

    agent = TD3(
        "mlp", env, lr_value=lr_value, lr_policy=lr_policy, replay_size=replay_size
    )
    trainer = OffPolicyTrainer(
        agent,
        env,
        log_interval=5,
        epochs=100,
        max_timesteps=16500,
        evaluate_episodes=10,
        max_ep_len=max_ep_len,
    )
    trainer.train()

    episode = 0
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

        state = next_state

        if np.any(done):
            for i, di in enumerate(done):
                if di:
                    episode += 1
                    episode_rewards.append(trainer.env.episode_reward[i])
                    trainer.env.episode_reward[i] = 0

        if episode == trainer.evaluate_episodes:
            eval_reward = float(np.mean(episode_rewards))

            trial.report(eval_reward, int(episode))
            break

    return eval_reward


study = optuna.create_study(
    study_name="1",
    direction="maximize",
    storage="sqlite:///td3--pendulum-v0--replay_size-max_ep_len-lr_value-lr_policy.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=20)
df = study.trials_dataframe(attrs=("number", "value", "params"))
df.to_pickle("logs/optuna_logs.pkl")

print("Best trial: ")
for key, value in study.best_trial.__dict__.items():
    print("{}: {}".format(key, value))
print("Eval Reward: ", study.best_value)
print("Params: ", study.best_params)
