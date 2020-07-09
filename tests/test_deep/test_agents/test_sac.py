import shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_sac(self):
    env = VectorEnv("Pendulum-v0", 2)
    algo = SAC("mlp", env, layers=[1, 1])

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
