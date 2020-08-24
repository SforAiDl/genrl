import shutil

from genrl.deep.agents.dqn import (
    DQN,
    CategoricalDQN,
    DoubleDQN,
    DuelingDQN,
    NoisyDQN,
    PrioritizedReplayDQN,
)
from genrl.deep.common import OffPolicyTrainer
from genrl.deep.common.buffers import PrioritizedBuffer
from genrl.deep.common.values import (
    MlpCategoricalValue,
    MlpDuelingValue,
    MlpNoisyValue,
    MlpValue,
)
from genrl.environments import VectorEnv


class TestDQN:
    def test_vanilla_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DQN("mlp", env, replay_size=100)
        assert isinstance(algo.model, MlpValue)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_double_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DoubleDQN("mlp", env, replay_size=100)
        assert isinstance(algo.model, MlpValue)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_dueling_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = DuelingDQN("mlp", env, replay_size=100)
        assert algo.dqn_type == "dueling"
        assert isinstance(algo.model, MlpDuelingValue)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_prioritized_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = PrioritizedReplayDQN("mlp", env, replay_size=100)
        assert isinstance(algo.model, MlpValue)
        assert isinstance(algo.replay_buffer, PrioritizedBuffer)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_noisy_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = NoisyDQN("mlp", env, replay_size=100)
        assert algo.dqn_type == "noisy"
        assert algo.noisy
        assert isinstance(algo.model, MlpNoisyValue)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_categorical_dqn(self):
        env = VectorEnv("CartPole-v0")
        algo = CategoricalDQN("mlp", env, replay_size=100)
        assert algo.dqn_type == "categorical"
        assert algo.noisy
        assert isinstance(algo.model, MlpCategoricalValue)
        trainer = OffPolicyTrainer(
            algo, env, log_mode=["csv"], logdir="./logs", max_ep_len=200, epochs=4
        )
        trainer.train()
        shutil.rmtree("./logs")
