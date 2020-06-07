import shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import VectorEnv


def test_gym_env():
    """
    Tests working of Gym Wrapper and the GymEnv function
    """
    env = VectorEnv("Pendulum-v0", env_type="gym")
    algo = SAC("mlp", env, layers=[1, 1])

    trainer = OffPolicyTrainer(algo, epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
