from genrl import VPG
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv

env = VectorEnv("Pong-v0", env_type="atari")
agent = VPG("cnn", env, rollout_size=6000)
trainer = OnPolicyTrainer(
    agent, env, log_interval=1, epochs=100, log_mode=["stdout"], device="cuda"
)

trainer.train()
trainer.evaluate()
