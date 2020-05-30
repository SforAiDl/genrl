import gym

from genrl.deep.common import OffPolicyTrainer, OnPolicyTrainer, Logger, venv
from genrl import PPO1, TD3


def test_on_policy_trainer():
    logger = Logger()
    env = venv("CartPole-v1", 1)
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert trainer.off_policy == False
    trainer.train()


def test_off_policy_trainer():
    env = venv("Pendulum-v0", 1)
    algo = TD3("mlp", env)
    trainer = OffPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert trainer.off_policy == True
    trainer.train()
