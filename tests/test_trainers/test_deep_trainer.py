import os
from shutil import rmtree

from genrl.agents import DDPG, PPO1
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer, OnPolicyTrainer


class TestDeepTrainer:
    def test_on_policy_trainer(self):
        env = VectorEnv("CartPole-v1", 2)
        algo = PPO1("mlp", env, rollout_size=128)
        trainer = OnPolicyTrainer(
            algo, env, ["stdout"], epochs=2, evaluate_episodes=2, max_timesteps=300
        )
        assert not trainer.off_policy
        trainer.train()
        trainer.evaluate()

    def test_off_policy_trainer(self):
        env = VectorEnv("Pendulum-v0", 2)
        algo = DDPG("mlp", env, replay_size=100)
        trainer = OffPolicyTrainer(
            algo,
            env,
            ["stdout"],
            epochs=2,
            evaluate_episodes=2,
            max_ep_len=300,
            max_timesteps=300,
        )
        assert trainer.off_policy
        trainer.train()
        trainer.evaluate()

    def test_save_params(self):
        """
        test saving algorithm state dict
        """
        env = VectorEnv("CartPole-v0", 1)
        algo = PPO1("mlp", env)
        trainer = OnPolicyTrainer(
            algo, env, ["stdout"], save_model="test_ckpt", save_interval=1, epochs=1
        )
        trainer.train()

        assert len(os.listdir("test_ckpt/PPO1_CartPole-v0")) != 0

    def test_load_params(self):
        """
        test loading algorithm parameters
        """
        env = VectorEnv("CartPole-v0", 1)
        algo = PPO1("mlp", env)
        trainer = OnPolicyTrainer(
            algo,
            env,
            epochs=0,
            load_hyperparams="test_ckpt/PPO1_CartPole-v0/0-log-0.toml",
            load_weights="test_ckpt/PPO1_CartPole-v0/0-log-0.pt",
        )
        trainer.train()

        rmtree("logs")
