from genrl.agents import A2C, DQN, DoubleDQN
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer, OnPolicyTrainer

env = VectorEnv("CartPole-v0", n_envs=5)
agent = DQN("mlp", env, replay_size=200)
trainer = OffPolicyTrainer(agent, env, epochs=50, max_timesteps=5e4, log_interval=25)
trainer.train()
trainer.evaluate()
