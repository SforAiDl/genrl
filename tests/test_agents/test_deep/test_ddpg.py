import shutil

from genrl.agents import DDPG
from genrl.core import NormalActionNoise
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer


class TestDDPG:
    def test_ddpg(self):
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

    def test_ddpg_shared(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = DDPG(
            "mlp",
            env,
            batch_size=5,
            noise=NormalActionNoise,
            shared_layers=[1, 1],
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
