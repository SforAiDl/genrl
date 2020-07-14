import shutil

from genrl import VPG
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv


def test_vpg():
    env = VectorEnv("CartPole-v0", 1)
    algo = VPG("mlp", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
