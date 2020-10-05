import gym

from genrl.agents.modelbased.cem.cem import CEM
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer


def test_CEM():
    env = VectorEnv("CartPole-v0", 1)
    algo = CEM(env, percentile=70, policy_layers=[100])
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
