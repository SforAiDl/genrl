import shutil

from genrl import DQN
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_dqn_cnn():
    env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

    # DQN
    algo = DQN("cnn", env)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")

    # Double DQN with prioritized replay buffer
    algo1 = DQN("cnn", env, double_dqn=True, prioritized_replay=True)

    trainer = OffPolicyTrainer(
        algo1, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")

    # Noisy DQN
    algo2 = DQN("cnn", env, noisy_dqn=True)

    trainer = OffPolicyTrainer(
        algo2, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")

    # Dueling DDQN
    algo3 = DQN("cnn", env, dueling_dqn=True, double_dqn=True)

    trainer = OffPolicyTrainer(
        algo3, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")

    # Categorical DQN
    algo4 = DQN("cnn", env, categorical_dqn=True)

    trainer = OffPolicyTrainer(
        algo4, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")
