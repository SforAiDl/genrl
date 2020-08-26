import shutil

from genrl.agents import DDPG
from genrl.core import NormalActionNoise
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer


def test_ddpg():
    env = VectorEnv("Pendulum-v0", 2)
    algo = DDPG(
        "mlp",
        env,
        batch_size=5,
        noise=NormalActionNoise,
        policy_layers=[1, 1],
        value_layers=[1, 1],
    )

    trainer = OffPolicyTrainer(
        algo,
        env,
        log_mode=["csv"],
        logdir="./logs",
        epochs=4,
        max_ep_len=200,
        warmup_steps=10,
        start_update=10,
    )
    trainer.train()
    shutil.rmtree("./logs")
