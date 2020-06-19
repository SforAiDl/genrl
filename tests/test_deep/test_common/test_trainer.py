from genrl import PPO1, TD3
from genrl.deep.common import OffPolicyTrainer, OnPolicyTrainer
from genrl.environments import VectorEnv


def test_on_policy_trainer():
    env = VectorEnv("CartPole-v1", 2)
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert not trainer.off_policy
    trainer.train()


def test_off_policy_trainer():
    env = VectorEnv("Pendulum-v0", 2)
    algo = TD3("mlp", env)
    trainer = OffPolicyTrainer(algo, env, ["stdout"], epochs=1)
    assert trainer.off_policy
    trainer.train()
