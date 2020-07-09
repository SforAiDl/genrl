import shutil

from genrl import DQN
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_dqn(self):
    env = VectorEnv("CartPole-v0", 2)
    # DQN
    algo = DQN("mlp", env)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")

    # Double DQN with prioritized replay buffer
    algo1 = DQN("mlp", env, double_dqn=True, prioritized_replay=True)

    trainer = OffPolicyTrainer(algo1, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")

    # Noisy DQN
    algo2 = DQN("mlp", env, noisy_dqn=True)

    trainer = OffPolicyTrainer(algo2, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")

    # Dueling DDQN
    algo3 = DQN("mlp", env, dueling_dqn=True, double_dqn=True)

    trainer = OffPolicyTrainer(algo3, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")

    # Categorical DQN
    algo4 = DQN("mlp", env, categorical_dqn=True)

    trainer = OffPolicyTrainer(algo4, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
