import gym

from genrl import PPO1, TD3
from genrl.deep.common import Logger, OffPolicyTrainer, OnPolicyTrainer, venv


def test_on_policy_trainer():
    logger = Logger()
    env = venv("CartPole-v1", 2)
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert trainer.off_policy is False
    trainer.train()


def test_off_policy_trainer():
    env = venv("Pendulum-v0", 2)
    algo = TD3("mlp", env)
    trainer = OffPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert trainer.off_policy is True
    trainer.train()
