from genrl.agents import BCQ, DDPG
from genrl.environments import VectorEnv
from genrl.trainers import OfflineTrainer, OffPolicyTrainer

env = VectorEnv("Pendulum-v0")
# agent = DDPG(
#     "mlp",
#     env,
#     replay_size=500,
# )
# trainer = OffPolicyTrainer(
#     agent,
#     env,
#     epochs=40,
#     log_interval=5,
#     max_timesteps=50000,
#     save_interval=2000,
# )
# trainer.train()
# trainer.evaluate()
# trainer.save(2000, True)

agent = BCQ(
    "mlp",
    env,
    replay_size=500,
)
trainer = OfflineTrainer(
    agent,
    env,
    log_interval=50,
    buffer_path="checkpoints/DDPG_Pendulum-v0/6-buffer-2000.pt",
    max_timesteps=1000,
)
trainer.train()
trainer.evaluate()
