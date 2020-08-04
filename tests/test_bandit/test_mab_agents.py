from genrl.bandit import (
    BayesianUCBMABAgent,
    BernoulliMAB,
    EpsGreedyMABAgent,
    GaussianMAB,
    GradientMABAgent,
    ThompsonSamplingMABAgent,
    UCBMABAgent,
)


class TestMABAgent:
    def test_eps_greedy_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = EpsGreedyMABAgent(bandit)
        policy.learn(10)

    def test_eps_greedy_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = EpsGreedyMABAgent(bandit)
        policy.learn(10)

    def test_ucb_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = UCBMABAgent(bandit)
        policy.learn(10)

    def test_ucb_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = UCBMABAgent(bandit)
        policy.learn(10)

    def test_gradient_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = GradientMABAgent(bandit)
        policy.learn(10)

    def test_gradient_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = GradientMABAgent(bandit)
        policy.learn(10)

    def test_bayesian_ucb_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = BayesianUCBMABAgent(bandit)
        policy.learn(10)

    def test_bayesian_ucb_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = BayesianUCBMABAgent(bandit)
        policy.learn(10)

    def test_thompson_sampling_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = ThompsonSamplingMABAgent(bandit)
        policy.learn(10)
