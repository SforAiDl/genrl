import shutil
from collections import OrderedDict

import numpy as np
import torch
from gym import GoalEnv, spaces

from genrl.agents import DQN
from genrl.core import ReplayBuffer
from genrl.core.buffers import HERWrapper
from genrl.environments.her_wrapper import HERGoalEnvWrapper
from genrl.trainers import OffPolicyTrainer
from genrl.trainers.her_trainer import HERTrainer


class BitFlippingEnv(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.
    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: (int) Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: (bool) Whether to use the discrete observation
        version or not, by default, it uses the MultiBinary one

    Adopted from Stable Baselines
    """

    def __init__(
        self, n_bits=10, continuous=False, max_steps=None, discrete_obs_space=False
    ):
        super(BitFlippingEnv, self).__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # In the discrete case, the agent act on the binary
            # representation of the observation replay
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2 ** n_bits - 1),
                    "achieved_goal": spaces.Discrete(2 ** n_bits - 1),
                    "desired_goal": spaces.Discrete(2 ** n_bits - 1),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "achieved_goal": spaces.MultiBinary(n_bits),
                    "desired_goal": spaces.MultiBinary(n_bits),
                }
            )

        self.obs_space = spaces.MultiBinary(n_bits)
        self.n_envs = 3
        self.episode_reward = torch.zeros(1)
        self.obs_dim = n_bits
        self.goal_dim = n_bits

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.state = None
        self.desired_goal = np.ones((n_bits,))
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()
        self.keys = ["observation", "achieved_goal", "desired_goal"]

    def convert_if_needed(self, state):
        """
        Convert to discrete space if needed.
        :param state: (np.ndarray)
        :return: (np.ndarray or int)
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum([state[i] * 2 ** i for i in range(len(state))]))
        return state

    def _get_obs(self):
        """
        Helper to create the observation.
        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict(
            [
                ("observation", self.convert_if_needed(self.state.copy())),
                ("achieved_goal", self.convert_if_needed(self.state.copy())),
                ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
            ]
        )

    def convert_dict_to_obs(self, obs_dict):
        return torch.as_tensor(
            np.concatenate([obs_dict[key] for key in self.keys]), dtype=torch.float32
        )

    def convert_obs_to_dict(self, obs):
        return OrderedDict(
            [
                ("observation", obs[: self.obs_dim]),
                ("achieved_goal", obs[self.obs_dim : self.obs_dim + self.goal_dim]),
                ("desired_goal", obs[self.obs_dim + self.goal_dim :]),
            ]
        )

    def reset(self):
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def sample(self):
        return torch.as_tensor(self.action_space.sample())

    def step(self, action):
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {"done": done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, _info
    ) -> float:
        # Deceptive reward: it is positive only when the goal is achieved
        if self.discrete_obs_space:
            return 0.0 if achieved_goal == desired_goal else -1.0
        return 0.0 if (achieved_goal == desired_goal).all() else -1.0

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self):
        pass


def test_her():
    env = BitFlippingEnv()
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
