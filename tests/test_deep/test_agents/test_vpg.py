import shutil

from genrl import VPG
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


def test_vpg():
    env = VectorEnv("CartPole-v0")
    algo = VPG("mlp", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_vpg_cnn():
    env = VectorEnv("Pong-v0", env_type="atari")
    algo = VPG("cnn", env, rollout_size=128)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
