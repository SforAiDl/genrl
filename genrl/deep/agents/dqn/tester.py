from genrl import OffPolicyTrainer, VectorEnv, DQN

env = VectorEnv("CartPole-v0", 1)
agent = DQN("mlp", env)
trainer = OffPolicyTrainer(agent, epochs=1)
trainer.train()
