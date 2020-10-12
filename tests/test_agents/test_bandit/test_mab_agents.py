from genrl.agents import (
    BayesianUCBMABAgent,
    BernoulliMAB,
    EpsGreedyMABAgent,
    GaussianMAB,
    GradientMABAgent,
    ThompsonSamplingMABAgent,
    UCBMABAgent,
)
from genrl.trainers import MABTrainer


class TestMABAgent:
    def test_eps_greedy_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = EpsGreedyMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_eps_greedy_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = EpsGreedyMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_ucb_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = UCBMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_ucb_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = UCBMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_gradient_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = GradientMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_gradient_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = GradientMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_bayesian_ucb_gaussian(self) -> None:
        bandit = GaussianMAB(arms=10, context_type="int")
        policy = BayesianUCBMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_bayesian_ucb_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = BayesianUCBMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)

    def test_thompson_sampling_bernoulli(self) -> None:
        bandit = BernoulliMAB(arms=10, context_type="int")
        policy = ThompsonSamplingMABAgent(bandit)
        trainer = MABTrainer(policy, bandit)
        trainer.train(timesteps=10)
