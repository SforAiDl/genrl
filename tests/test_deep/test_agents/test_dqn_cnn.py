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


def test_double_dqn_cnn():
    env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

    # Double DQN with prioritized replay buffer
    algo = DQN("cnn", env, double_dqn=True, prioritized_replay=True)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")


def test_noisy_dqn_cnn():
    env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

    # Noisy DQN
    algo = DQN("cnn", env, noisy_dqn=True)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")


def test_dueling_dqn_cnn():
    env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

    # Dueling DDQN
    algo = DQN("cnn", env, dueling_dqn=True, double_dqn=True)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")


def test_categorical_dqn_cnn():
    env = VectorEnv("Pong-v0", n_envs=2, env_type="atari")

    # Categorical DQN
    algo = DQN("cnn", env, categorical_dqn=True)

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, steps_per_epoch=200
    )
    trainer.train()
    shutil.rmtree("./logs")
