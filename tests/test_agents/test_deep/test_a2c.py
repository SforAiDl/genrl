import shutil

from genrl.agents import A2C
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


class TestA2C:
    def test_a2c_discrete(self):
        env = VectorEnv("CartPole-v0", 1)
        algo = A2C("mlp", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_a2c_continuous(self):
        env = VectorEnv("Pendulum-v0", 1)
        algo = A2C("mlp", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        trainer.evaluate()
        shutil.rmtree("./logs")

    def test_a2c_cnn(self):
        env = VectorEnv("Pong-v0", 1, env_type="atari")
        algo = A2C("cnn", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_a2c_shared_discrete(self):
        env = VectorEnv("CartPole-v0", 1)
        algo = A2C("mlp", env, shared_layers=(32, 32), rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")
