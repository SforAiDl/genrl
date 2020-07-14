from genrl import DDPG, PPO1
from genrl.deep.common import OffPolicyTrainer, OnPolicyTrainer
from genrl.environments import VectorEnv


def test_on_policy_trainer():
    env = VectorEnv("CartPole-v1", 2)
    algo = PPO1("mlp", env)
    trainer = OnPolicyTrainer(algo, env, ["stdout"], epochs=1, evaluate_episodes=2)
    assert not trainer.off_policy
    trainer.train()
    trainer.evaluate()


def test_off_policy_trainer():
    env = VectorEnv("Pendulum-v0", 2)
    algo = DDPG("mlp", env, replay_size=100)
    trainer = OffPolicyTrainer(algo, env, ["stdout"], epochs=1, evaluate_episodes=2)
    assert trainer.off_policy
    trainer.train()
    trainer.evaluate()


test_off_policy_trainer()
