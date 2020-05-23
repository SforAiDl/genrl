import gym, shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import GymEnv


def test_gym_env(self):
    """
    Tests working of Gym Wrapper and the GymEnv function
    """
    env = GymEnv("Pendulum-v0")
    algo = SAC("mlp", env, layers=[1, 1])

    trainer = OffPolicyTrainer(
        algo, env, log_mode=["csv"], logdir="./logs", epochs=1, render=False
    )
    trainer.train()
    shutil.rmtree("./logs")
