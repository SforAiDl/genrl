import shutil

from genrl.agents import CEM
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


class TestCEM:
    def test_CEM(self):
        env = VectorEnv("CartPole-v0", 1)
        algo = CEM(
            "mlp",
            env,
            percentile=70,
            policy_layers=[100],
            rollout_size=100,
            simulations_per_epoch=100,
        )
        trainer = OnPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", epochs=1
        )
        trainer.train()
        shutil.rmtree("./logs")
