import shutil

from genrl import SAC
from genrl.deep.common import OffPolicyTrainer
from genrl.environments import ClipAction, GymEnv, RescaleAction, VectorEnv


class TestWrappers:
    def test_gym_env(self):
        """
        Tests working of Gym Wrapper and the GymEnv function
        """
        env = VectorEnv("Pendulum-v0", env_type="gym")
        algo = SAC("mlp", env, layers=[1, 1])

        trainer = OffPolicyTrainer(algo, env, epochs=1)
        trainer.train()
        shutil.rmtree("./logs")

    def test_clip_action(self):
        """
        Tests working of Clip Action Wrapper
        """
        env = GymEnv("Pendulum-v0")
        env = ClipAction(env, 1.0, 2.0)

        action = env.action_space.sample()
        assert action == 1.0

    def test_rescale_action(self):
        """
        Tests working of Rescale Action Wrapper
        """
        env = GymEnv("Pendulum-v0")
        env = RescaleAction(env, 1, 5)

        action = env.action_space.sample()
        assert action >= 1.0
