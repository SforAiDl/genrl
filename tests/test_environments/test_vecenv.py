import numpy as np
import pytest

from genrl.environments.suite import VectorEnv
from genrl.environments.vec_env import RunningMeanStd, VecNormalize


class TestVecEnvs:
    def test_vecenv_parallel(self):
        env = VectorEnv("CartPole-v1", 2, parallel=True)
        env.seed(0)
        ob, ac = env.get_spaces()

        env.reset()
        env.step(env.sample())
        env.close()

    def test_vecenv_serial(self):
        env = VectorEnv("CartPole-v1", 2, parallel=False)
        env.seed(0)
        ob, ac = env.observation_spaces, env.action_spaces

        env.reset()
        env.step(env.sample())
        env.close()

    def test_vecnormalize(self):
        pass

    def test_rms(self):
        rms = RunningMeanStd(shape=(5, 2))
        batch = np.random.randn(5, 2)
        rms.update(batch)

        assert rms.mean.shape == (5, 2)
        assert rms.var.shape == (5, 2)
        assert rms.count == pytest.approx(5, 1e-4)
