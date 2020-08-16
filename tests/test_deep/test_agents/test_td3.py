import shutil

from genrl import TD3
from genrl.deep.common import OffPolicyTrainer, OrnsteinUhlenbeckActionNoise
from genrl.environments import VectorEnv


def test_td3():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3(
        "mlp", env, noise=OrnsteinUhlenbeckActionNoise, layers=[1, 1], replay_size=100
    )

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=5, max_ep_len=500
    )
    trainer.train()
    shutil.rmtree("./logs")
