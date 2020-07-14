import shutil

from genrl import A2C
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv


def test_a2c():
    env = VectorEnv("CartPole-v0", 1)
    algo = A2C("mlp", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
