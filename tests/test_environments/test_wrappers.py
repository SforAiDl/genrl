import shutil

import pytest

from genrl.environments import ClipAction, GymEnv, RescaleAction, VectorEnv
from genrl.trainers import OffPolicyTrainer


class TestWrappers:
    def test_gym_env(self):
        """
        Tests working of Gym Wrapper and the GymEnv function
        """
        env = VectorEnv("Pendulum-v0", env_type="gym")
        env.reset()
        env.step(env.sample())
        env.close()

    def test_clip_action(self):
        """
        Tests working of Clip Action Wrapper
        """
        env = GymEnv("Pendulum-v0")
        clip_env = ClipAction(GymEnv("Pendulum-v0"))

        env.seed(42)
        clip_env.seed(42)

        env.reset()
        clip_env.reset()

        actions = [[-1.2], [-10.5], [9.823], [0.564], [-0.1123], [1.1]]
        for action in actions:
            state, reward, done, _ = env.step(action)
            clip_state, clip_reward, clip_done, _ = clip_env.step(action)

            assert state == pytest.approx(clip_state)
            assert reward == pytest.approx(clip_reward)
            assert done == clip_done

    def test_rescale_action(self):
        """
        Tests working of Rescale Action Wrapper
        """
        env = GymEnv("Pendulum-v0")
        env = RescaleAction(env, 1, 5)

        action = env.action_space.sample()
        assert action >= 1.0
