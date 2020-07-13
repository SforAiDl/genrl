import shutil

from genrl import (
    BernoulliCB,
    BootstrapNeuralAgent,
    CovertypeDataBandit,
    FixedAgent,
    LinearPosteriorAgent,
    NeuralGreedyAgent,
    NeuralLinearPosteriorAgent,
    NeuralNoiseSamplingAgent,
    VariationalAgent,
)
from genrl.deep.common import BanditTrainer


class TestDCBAgents:
    def test_linear_posterior_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = LinearPosteriorAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_neural_linear_posterior_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = NeuralLinearPosteriorAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_neural_greedy_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = NeuralGreedyAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_variational_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = VariationalAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_bootstrap_neural_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = BootstrapNeuralAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_neural_noise_sampling_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = NeuralNoiseSamplingAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_fixed_agent(self) -> None:
        bandits = []
        bandits.append(CovertypeDataBandit(download=True))
        bandits.append(BernoulliCB(bandits=10, arms=10))
        for bandit in bandits:
            agent = FixedAgent(bandit)
            trainer = BanditTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")
