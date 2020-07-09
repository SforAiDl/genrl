import shutil

from genrl import PPO1
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv


def test_ppo1(self):
    env = VectorEnv("CartPole-v0", 1)
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
