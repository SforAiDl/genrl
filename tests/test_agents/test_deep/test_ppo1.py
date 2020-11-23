import shutil

from genrl.agents import PPO1
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


class TestPPO:
    def test_ppo1(self):
        env = VectorEnv("CartPole-v0")
        algo = PPO1("mlp", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_ppo1_continuous(self):
        env = VectorEnv("Pendulum-v0")
        algo = PPO1("mlp", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_ppo1_cnn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = PPO1("cnn", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_ppo1_shared(self):
        env = VectorEnv("CartPole-v0")
        algo = PPO1("mlp", env, shared_layers=[32, 32], rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")
