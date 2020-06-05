import shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import GymEnv


def test_gym_env():
    """
    Tests working of Gym Wrapper and the GymEnv function
    """
    env = GymEnv("Pendulum-v0")
    algo = SAC("mlp", env, layers=[1, 1])

    trainer = OffPolicyTrainer(algo, epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
