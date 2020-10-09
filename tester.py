# %%
from genrl.agents import BCQ
from genrl.environments import VectorEnv
from genrl.trainers import OfflineTrainer

# %%
env = VectorEnv("Pendulum-v0")
agent = BCQ(
    "mlp",
    env,
    replay_size=100,
)
trainer = OfflineTrainer(
    agent,
    env,
    epochs=10,
    log_interval=1,
    log_mode=["stdout"],
    buffer_path="saves/DDPG_Pendulum-v0/run-0.pt",
    max_timesteps=100000,
)

# %%
trainer.train()
