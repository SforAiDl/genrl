from genrl import (
    GaussianBandit,
    BernoulliBandit,
    EpsGreedyPolicy,
    UCBPolicy,
    SoftmaxActionSelectionPolicy,
    BayesianUCBPolicy,
    ThompsonSamplingPolicy,
)

class TestBandit:
    def test_eps_greedy_gaussian(self):
        bandit = GaussianBandit(arms=10)
        policy = EpsGreedyPolicy(bandit)
        policy.learn(10)

    def test_ucb_gaussian(self):
        bandit = GaussianBandit(arms=10)
        policy = UCBPolicy(bandit)
        policy.learn(10)
    
    def test_softmax_gaussian(self):
        bandit = GaussianBandit(arms=10)
        policy = SoftmaxActionSelectionPolicy(bandit)
        policy.learn(10)
    
    def test_eps_greedy_bernoulli(self):
        bandit = BernoulliBandit(arms=10)
        policy = EpsGreedyPolicy(bandit)
        policy.learn(10)

    def test_ucb_bernoulli(self):
        bandit = BernoulliBandit(arms=10)
        policy = UCBPolicy(bandit)
        policy.learn(10)

    def test_bayesian_ucb_bernoulli(self):
        bandit = BernoulliBandit(arms=10)
        policy = BayesianUCBPolicy(bandit)
        policy.learn(10)

    def test_thompson_sampling_bernoulli(self):
        bandit = BernoulliBandit(arms=10)
        policy = ThompsonSamplingPolicy(bandit)
        policy.learn(10)
