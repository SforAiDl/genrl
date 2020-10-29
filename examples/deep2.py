from genrl.agents import DDPG
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer

env = VectorEnv("Pendulum-v0", n_envs=1)
agent = DDPG("mlp", env)
trainer = OffPolicyTrainer(agent, env, max_timesteps=20000)
trainer.train()
trainer.evaluate()
