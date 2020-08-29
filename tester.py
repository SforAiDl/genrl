from genrl.agents import DQN
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer

env = VectorEnv("CartPole-v0")
agent = DQN("mlp", env)
trainer = OffPolicyTrainer(
    agent, env, log_interval=10, log_mode=["stdout", "tensorboard"]
)
trainer.train()
trainer.evaluate()
