import shutil

from genrl.deep.agents import DQN, DoubleDQN, DuelingDQN, PrioritizedReplayDQN
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


class TestDQN:
    def test_vanilla_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DQN("mlp", env)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_double_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DoubleDQN("mlp", env)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_dueling_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DuelingDQN("mlp", env)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    # def test_prioritized_dqn(self):
    #     env = VectorEnv("CartPole-v0")
    #     algo = PrioritizedReplayDQN("mlp", env)
    #     trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    #     trainer.train()
    #     shutil.rmtree("./logs")
