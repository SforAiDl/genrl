from genrl import (
    BayesianUCBCBPolicy,
    BayesianUCBPolicy,
    BernoulliBandit,
    BernoulliCB,
    CovertypeDataBandit,
    EpsGreedyCBPolicy,
    EpsGreedyPolicy,
    GaussianBandit,
    GaussianCB,
    GradientCBPolicy,
    GradientPolicy,
    LinearPosteriorAgent,
    MushroomDataBandit,
    NeuralGreedyAgent,
    NeuralLinearPosteriorAgent,
    StatlogDataBandit,
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
        policy = GradientPolicy(bandit)
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
        policy = GradientCBPolicy(bandit)
        policy.learn(10)

    def test_bayesian_ucb_bernoulli_cb(self) -> None:
        bandit = BernoulliCB(bandits=10, arms=10)
        policy = BayesianUCBCBPolicy(bandit)
        policy.learn(10)

    def test_thompson_bernoulli_cb(self) -> None:
        bandit = BernoulliCB(bandits=10, arms=10)
        policy = ThompsonSamplingCBPolicy(bandit)
        policy.learn(10)

    def test_covertype_data_bandit(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_mushroom_data_bandit(self) -> None:
        bandit = MushroomDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_statlog_data_bandit(self) -> None:
        bandit = StatlogDataBandit(download=True)
        _ = bandit.reset()
        for i in range(bandit.n_actions):
            _, _ = bandit.step(i)

    def test_linear_posterior_agent(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        policy = LinearPosteriorAgent(bandit)
        policy.learn(10)

    def test_neural_linear_posterior_agent(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        policy = NeuralLinearPosteriorAgent(bandit)
        policy.learn(10)

    def test_neural_greedy_agent(self) -> None:
        bandit = CovertypeDataBandit(download=True)
        policy = NeuralGreedyAgent(bandit)
        policy.learn(10)
