from jigglypuffRL import (
    SoftmaxActionSelection,
    UCBBernoulliBandit,
    UCBGaussianBandit,
    EpsGreedyBernoulliBandit,
    EpsGreedyGaussianBandit,
    ThompsonSampling,
    BayesianUCBBernoulliBandit,
)

class TestBandit:
    def test_softmax(self):
        softmaxBandit = SoftmaxActionSelection(1, 10)
        softmaxBandit.learn(10)

    def test_ucb_gaussian(self):
        ucbBandit = UCBGaussianBandit(1, 10)
        ucbBandit.learn(10)
        
    def test_eps_gaussian(self):
        epsGreedyBandit = EpsGreedyGaussianBandit(1, 10, 0.05)
        epsGreedyBandit.learn(10)

    def test_eps_bernoulli(self):
        epsbernoulli = EpsGreedyBernoulliBandit(1, 10, 0.05)
        epsbernoulli.learn(10)

    def test_ucb_bernoulli(self):
        ucbbernoulli = UCBBernoulliBandit(1, 10)
        ucbbernoulli.learn(10)

    def test_thompson(self):
        thsampling = ThompsonSampling(1, 10)
        thsampling.learn(10)

    def test_bayesian(self):
        bayesianbandit = BayesianUCBBernoulliBandit(1, 10)
        bayesianbandit.learn(10)

    