from genrl import (
    BayesianUCBCBPolicy,
    BayesianUCBPolicy,
    BernoulliBandit,
    BernoulliCB,
    EpsGreedyCBPolicy,
    EpsGreedyPolicy,
    GaussianBandit,
    GaussianCB,
    GradientBasedCBPolicy,
    GradientBasedPolicy,
    ThompsonSamplingCBPolicy,
    ThompsonSamplingPolicy,
    UCBCBPolicy,
    UCBPolicy,
)


class TestBandit:
    def test_eps_greedy_gaussian(self) -> None:
        bandit = GaussianBandit(arms=10)
        policy = EpsGreedyPolicy(bandit)
        policy.learn(10)

    def test_ucb_gaussian(self) -> None:
        bandit = GaussianBandit(arms=10)
        policy = UCBPolicy(bandit)
        policy.learn(10)

    def test_gradient_gaussian(self) -> None:
        bandit = GaussianBandit(arms=10)
        policy = GradientBasedPolicy(bandit)
        policy.learn(10)

    def test_eps_greedy_bernoulli(self) -> None:
        bandit = BernoulliBandit(arms=10)
        policy = EpsGreedyPolicy(bandit)
        policy.learn(10)

    def test_ucb_bernoulli(self) -> None:
        bandit = BernoulliBandit(arms=10)
        policy = UCBPolicy(bandit)
        policy.learn(10)

    def test_bayesian_ucb_bernoulli(self) -> None:
        bandit = BernoulliBandit(arms=10)
        policy = BayesianUCBPolicy(bandit)
        policy.learn(10)

    def test_thompson_sampling_bernoulli(self) -> None:
        bandit = BernoulliBandit(arms=10)
        policy = ThompsonSamplingPolicy(bandit)
        policy.learn(10)

    def test_eps_greedy_gaussian_cb(self) -> None:
        bandit = GaussianCB(bandits=10, arms=10)
        policy = EpsGreedyCBPolicy(bandit)
        policy.learn(10)

    def test_ucb_bernoulli_cb(self) -> None:
        bandit = BernoulliCB(bandits=10, arms=10)
        policy = UCBCBPolicy(bandit)
        policy.learn(10)

    def test_gradient_gaussian_cb(self) -> None:
        bandit = GaussianCB(bandits=10, arms=10)
        policy = GradientBasedCBPolicy(bandit)
        policy.learn(10)

    def test_bayesian_ucv_bernoulli_cb(self) -> None:
        bandit = BernoulliCB(bandits=10, arms=10)
        policy = BayesianUCBCBPolicy(bandit)
        policy.learn(10)

    def test_thompson_bernoulli_cb(self) -> None:
        bandit = BernoulliCB(bandits=10, arms=10)
        policy = ThompsonSamplingCBPolicy(bandit)
        policy.learn(10)
