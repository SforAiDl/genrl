import shutil

from genrl.agents import (
    BernoulliMAB,
    BootstrapNeuralAgent,
    FixedAgent,
    LinearPosteriorAgent,
    NeuralGreedyAgent,
    NeuralLinearPosteriorAgent,
    NeuralNoiseSamplingAgent,
    VariationalAgent,
)
from genrl.trainers import DCBTrainer
from genrl.utils import CovertypeDataBandit

from .utils import write_data


class TestCBAgent:
    def _test_fn(self, agent_class) -> None:
        bandits = []
        d = """2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
2590,56,2,212,-6,390,220,235,151,6225,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
2804,139,9,268,65,3180,234,238,135,6121,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2
2785,155,18,242,118,3090,238,238,122,6211,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2
2595,45,2,153,-1,391,220,234,150,6172,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,5
"""
        fpath = write_data("covtype.data", d)
        bandits.append(CovertypeDataBandit(path=fpath))
        fpath.unlink()
        bandits.append(BernoulliMAB(bandits=10, arms=10))
        for bandit in bandits:
            agent = agent_class(bandit)
            trainer = DCBTrainer(agent, bandit, log_mode=["stdout"])
            trainer.train(timesteps=10, update_interval=2, update_after=5, batch_size=2)
            shutil.rmtree("./logs")

    def test_linear_posterior_agent(self) -> None:
        self._test_fn(LinearPosteriorAgent)

    def test_neural_linear_posterior_agent(self) -> None:
        self._test_fn(NeuralLinearPosteriorAgent)

    def test_neural_greedy_agent(self) -> None:
        self._test_fn(NeuralGreedyAgent)

    def test_variational_agent(self) -> None:
        self._test_fn(VariationalAgent)

    def test_bootstrap_neural_agent(self) -> None:
        self._test_fn(BootstrapNeuralAgent)

    def test_neural_noise_sampling_agent(self) -> None:
        self._test_fn(NeuralNoiseSamplingAgent)

    def test_fixed_agent(self) -> None:
        self._test_fn(FixedAgent)
