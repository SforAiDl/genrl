import shutil

from genrl.deep.agents import (
    DQN,
    CategoricalDQN,
    DoubleDQN,
    DuelingDQN,
    NoisyDQN,
    PrioritizedReplayDQN,
)
from genrl.deep.common import OffPolicyTrainer
from genrl.deep.common.values import (
    CnnCategoricalValue,
    CnnDuelingValue,
    CnnNoisyValue,
    CnnValue,
)
from genrl.environments import VectorEnv


class TestDQNCNN:
    def test_vanilla_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DQN("cnn", env, batch_size=5, replay_size=100)
        assert isinstance(algo.model, CnnValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_double_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DoubleDQN("cnn", env, batch_size=5, replay_size=100)
        assert isinstance(algo.model, CnnValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_dueling_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DuelingDQN("cnn", env, batch_size=5, replay_size=100)
        assert algo.dqn_type == "dueling"
        assert isinstance(algo.model, CnnDuelingValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_prioritized_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = PrioritizedReplayDQN("cnn", env, batch_size=5, replay_size=100)
        assert isinstance(algo.model, CnnValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_noisy_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = NoisyDQN("cnn", env, batch_size=5, replay_size=100)
        assert algo.dqn_type == "noisy"
        assert algo.noisy
        assert isinstance(algo.model, CnnNoisyValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_categorical_dqn(self):
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = CategoricalDQN("cnn", env, batch_size=5, replay_size=100)
        assert algo.dqn_type == "categorical"
        assert algo.noisy
        assert isinstance(algo.model, CnnCategoricalValue)
        trainer = OffPolicyTrainer(
            algo,
            env,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=200,
            epochs=4,
            warmup_steps=10,
            start_update=10,
        )
        trainer.train()
        shutil.rmtree("./logs")
