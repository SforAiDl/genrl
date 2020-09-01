import pytest
import torch

from genrl.environments.suite import GymEnv, VectorEnv
from genrl.environments.vec_env import RunningMeanStd, VecMonitor, VecNormalize


class TestVecEnvs:
    def test_vecenv_parallel(self):
        """
        Tests working of parallel VecEnvs
        """
        env = VectorEnv("CartPole-v1", 2, parallel=True)
        env.seed(0)
        observation_space, action_space = env.get_spaces()

        env.reset()
        env.step(env.sample())
        env.close()

    def test_vecenv_serial(self):
        """
        Tests working of serial VecEnvs
        """
        env = VectorEnv("CartPole-v1", 2, parallel=False)
        env.seed(0)
        observation_space, action_space = env.get_spaces()

        env.reset()
        env.step(env.sample())
        env.close()

    def test_vecnormalize(self):
        """
        Tests working of the VecNormalize wrapper
        """
        env = VectorEnv("CartPole-v1", 2)
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_reward=1.0,
        )
        env.reset()
        _, rewards, _, _ = env.step(env.sample())
        env.close()

        assert (-1.0 <= rewards).byte().all()
        assert (1.0 >= rewards).byte().all()

    def test_vecmonitor(self):
        """
        Tests working of the VecMonitor wrapper
        """
        env = VectorEnv("CartPole-v1", 2)
        env = VecMonitor(env, history_length=1)

        env.reset()
        _, _, _, info = env.step(env.sample())
        env.close()

        dones = [0, 0]
        while not dones[0]:
            _, _, dones, infos = env.step(env.sample())

        info = infos[0]["episode"]

        assert info["Episode Rewards"]
        assert info["Episode Length"]
        assert info["Time taken"]

    def test_rms(self):
        """
        Tests working of the RMS utility function
        """
        rms = RunningMeanStd(shape=(5, 2))
        batch = torch.randn(5, 2)
        rms.update(batch)

        assert rms.mean.shape == (5, 2)
        assert rms.var.shape == (5, 2)
        assert rms.count == pytest.approx(5, 1e-4)
