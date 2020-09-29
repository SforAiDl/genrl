import shutil

from genrl.agents import A2C
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


def test_a2c():
    env = VectorEnv("CartPole-v0", 1)
    algo = A2C("mlp", env, rollout_size=128)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_a2c_cnn():
    env = VectorEnv("Pong-v0", 1, env_type="atari")
    algo = A2C("cnn", env, rollout_size=128)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_a2c_shared():
    env = VectorEnv("CartPole-v0", 1)
    algo = A2C("mlp", env, shared_layers=(32, 32), rollout_size=128)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
