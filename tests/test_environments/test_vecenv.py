from genrl.environments.suite import VectorEnv


def test_vecenv_parallel():
    env = VectorEnv("CartPole-v1", 2, parallel=True)
    env.seed(0)
    ob, ac = env.get_spaces()

    env.reset()
    env.step(env.sample())
