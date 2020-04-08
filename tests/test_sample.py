from jigglypuffRL import EpsGreedyGaussianBandit


def test():
    bandit = EpsGreedyGaussianBandit(1, 10, 0.01)
    assert bandit.eps == 0.01
