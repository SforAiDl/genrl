import pytest
from genrl.deep.common import venv

def test_vecenv_parallel():
    env = venv("CartPole-v1",2,parallel=True)
    env.seed(0)
    ob, ac = env.get_spaces()

    ob = env.reset()
    env.step(env.sample())