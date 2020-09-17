import shutil

import gym
import highway_env
from gym.core import Wrapper

from genrl.agents import DDPG, SAC, TD3
from genrl.core import HERWrapper, ReplayBuffer
from genrl.environments import HERGoalEnvWrapper, VectorEnv
from genrl.trainers import HERTrainer


class TestHER:
    def _test_agent(self, agent):
        env = gym.make("parking-v0")
        env = HERGoalEnvWrapper(env)
        algo = agent("mlp", env, batch_size=10, policy_layers=[1], value_layers=[1])
        buffer = HERWrapper(ReplayBuffer(10), 1, "future", env)
        trainer = HERTrainer(
            algo,
            env,
            buffer=buffer,
            log_mode=["csv"],
            logdir="./logs",
            max_ep_len=10,
            epochs=int(1),
            warmup_steps=1,
            start_update=1,
        )
        trainer.train()
        shutil.rmtree("./logs")

    def test_DDPG(self):
        self._test_agent(DDPG)

    def test_SAC(self):
        self._test_agent(SAC)

    def test_TD3(self):
        self._test_agent(TD3)
