import gym

from jigglypuffRL import Trainer, OffPolicyTrainer, OnPolicyTrainer, PPO1, TD3, Logger

def test_on_policy_trainer():
    logger = Logger()
    env = gym.make("CartPole-v1")
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, logger)
    assert trainer.off_policy == False
    trainer.train()

def test_off_policy_trainer():
    logger = Logger()
    env = gym.make("Pendulum-v0")
    algo = TD3("mlp", env)
    trainer = OnPolicyTrainer(algo, env, logger)
    assert trainer.off_policy == True
    trainer.train()
