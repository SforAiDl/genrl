import shutil

from genrl import TD3
from genrl.deep.common import OffPolicyTrainer, OrnsteinUhlenbeckActionNoise
from genrl.environments import VectorEnv


def test_td3():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3(
        "mlp",
        env,
        noise=OrnsteinUhlenbeckActionNoise,
        policy_layers=[1, 1],
        q1_layers=[1, 1],
        q2_layers=[1, 1],
    )

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
