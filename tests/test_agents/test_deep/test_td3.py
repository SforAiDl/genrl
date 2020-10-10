import shutil

from genrl.agents import TD3
from genrl.core import OrnsteinUhlenbeckActionNoise
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer


class TestTD3:
    def test_td3(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = TD3(
            "mlp",
            env,
            batch_size=5,
            noise=OrnsteinUhlenbeckActionNoise,
            policy_layers=[1, 1],
            value_layers=[1, 1],
        )

        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            epochs=5,
            max_ep_len=500,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_td3_shared(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = TD3(
            "mlp",
            env,
            batch_size=5,
            noise=OrnsteinUhlenbeckActionNoise,
            shared_layers=[1, 1],
            policy_layers=[1, 1],
            value_layers=[1, 1],
        )

        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            epochs=5,
            max_ep_len=500,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")
