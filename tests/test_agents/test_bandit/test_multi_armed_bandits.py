from genrl.agents import BernoulliMAB, GaussianMAB


class TestMultiArmedBandit:
    def test_gaussian_mab(self) -> None:
        bandit = GaussianMAB(arms=10)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_bernoulli_mab(self) -> None:
        bandit = BernoulliMAB(arms=10)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)
