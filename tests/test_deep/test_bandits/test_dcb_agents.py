from genrl import (
    BernoulliCB,
    BootstrapNeuralAgent,
    CovertypeDataBandit,
    LinearPosteriorAgent,
    NeuralGreedyAgent,
    NeuralLinearPosteriorAgent,
    NeuralNoiseSamplingAgent,
    VariationalAgent,
    FixedAgent,
)


class TestDCBAgents:
    def test_linear_posterior_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = LinearPosteriorAgent(bandit)
            policy.learn(10)

    def test_neural_linear_posterior_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = NeuralLinearPosteriorAgent(bandit)
            policy.learn(10)

    def test_neural_greedy_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = NeuralGreedyAgent(bandit)
            policy.learn(10)

    def test_variational_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = VariationalAgent(bandit)
            policy.learn(10)

    def test_bootstrap_neural_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = BootstrapNeuralAgent(bandit)
            policy.learn(10)

    def test_neural_noise_sampling_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = NeuralNoiseSamplingAgent(bandit)
            policy.learn(10)

    def test_fixed_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            policy = FixedAgent(bandit)
            policy.learn(10)
