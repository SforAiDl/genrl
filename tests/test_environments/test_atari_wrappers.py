import shutil

import gym

from genrl.agents import DQN
from genrl.environments import AtariEnv, AtariPreprocessing, FrameStack, VectorEnv
from genrl.trainers import OffPolicyTrainer


class TestAtari:
    def test_atari_preprocessing(self):
        """
        Tests Atari Preprocessing wrapper
        """
        env = gym.make("Pong-v0")
        atari_env = AtariPreprocessing(env)

        state = atari_env.reset()
        assert state.shape == (84, 84)
        action = atari_env.action_space.sample()
        state, reward, done, info = atari_env.step(action)
        assert state.shape == (84, 84)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        atari_env.close()

    def test_framestack(self):
        """
        Tests Frame Stack wrapper
        """
        env = gym.make("Pong-v0")
        atari_env = FrameStack(env)

        state = atari_env.reset()
        assert state.shape == (1, 4, 210, 160, 3)
        action = atari_env.action_space.sample()
        state, reward, done, info = atari_env.step(action)
        assert state.shape == (1, 4, 210, 160, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        atari_env.close()

    def test_atari_env(self):
        """
        Tests working of Atari Wrappers and the AtariEnv function
        """
        env = VectorEnv("Pong-v0", env_type="atari")
        algo = DQN("cnn", env, batch_size=5, replay_size=100, value_layers=[1, 1])

        trainer = OffPolicyTrainer(
            algo, env, epochs=5, max_ep_len=200, warmup_steps=10, start_update=10
        )
        trainer.train()
        shutil.rmtree("./logs")
