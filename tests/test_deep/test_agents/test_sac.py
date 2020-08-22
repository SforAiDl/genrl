import shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_sac():
    env = VectorEnv("Pendulum-v0", 2)
    algo = SAC("mlp", env, policy_layers=[1, 1], value_layers=[1, 1])

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=5, max_ep_len=500
    )
    trainer.train()
    shutil.rmtree("./logs")
