import shutil

from genrl.agents import DQN
from genrl.core import HERWrapper, ReplayBuffer
from genrl.environments import HERGoalEnvWrapper
from genrl.environments.custom_envs import BitFlipEnv, BitFlippingEnv
from genrl.trainers import HERTrainer


def test_her():
    env = BitFlippingEnv()
    env = HERGoalEnvWrapper(env)
    algo = DQN("mlp", env, batch_size=5, replay_size=10, value_layers=[1, 1])
    buffer = HERWrapper(ReplayBuffer(1000), 1, "future", env)
    print(isinstance(buffer, ReplayBuffer))
    trainer = HERTrainer(
        algo,
        env,
        buffer=buffer,
        log_mode=["csv"],
        logdir="./logs",
        max_ep_len=200,
        epochs=100,
        warmup_steps=10,
        start_update=10,
    )
    trainer.train()
    shutil.rmtree("./logs")
