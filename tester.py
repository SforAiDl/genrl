from genrl import A2C, PPO1, DQN  # noqa
from genrl.deep.common import OnPolicyTrainer, OffPolicyTrainer  # noqa
from genrl.environments import VectorEnv


env = VectorEnv("CartPole-v0", env_type="gym")
agent = DQN("mlp", env)
trainer = OffPolicyTrainer(agent, env, epochs=100, log_mode=["csv", "stdout"], logdir="logs", log_interval=50, device="cuda")
trainer.train()
trainer.evaluate()
