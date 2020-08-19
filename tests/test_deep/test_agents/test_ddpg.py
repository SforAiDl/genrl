import shutil

from genrl import DDPG
from genrl.deep.common import NormalActionNoise, OffPolicyTrainer
from genrl.environments import VectorEnv


def test_ddpg():
    env = VectorEnv("Pendulum-v0", 2)
    algo = DDPG(
        "mlp", env, noise=NormalActionNoise, policy_layers=[1, 1], value_layers=[1, 1]
    )

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=4, max_ep_len=200
    )
    trainer.train()
    shutil.rmtree("./logs")
