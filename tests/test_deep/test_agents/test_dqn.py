import shutil

from genrl import DQN
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_dqn():
    env = VectorEnv("CartPole-v0", 2)
    # DQN
    algo = DQN("mlp", env)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


# def test_double_dqn():
# Double DQN with prioritized replay buffer
# algo = DQN("mlp", env, double_dqn=True, prioritized_replay=True)

# trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
# trainer.train()
# shutil.rmtree("./logs")


def test_noisy_dqn():
    env = VectorEnv("CartPole-v0", 2)
    # Noisy DQN
    algo = DQN("mlp", env, noisy_dqn=True)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_dueling_dqn():
    env = VectorEnv("CartPole-v0", 2)
    # Dueling DDQN
    algo = DQN("mlp", env, dueling_dqn=True, double_dqn=True)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_categorical_dqn():
    env = VectorEnv("CartPole-v0", 2)
    # Categorical DQN
    algo = DQN("mlp", env, categorical_dqn=True)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
