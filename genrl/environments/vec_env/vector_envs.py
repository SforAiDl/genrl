import multiprocessing as mp
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Iterator, List, Tuple

import gym
import torch


def worker(parent_conn: mp.Pipe, child_conn: mp.Pipe, env: gym.Env):
    """
    Worker class to facilitate multiprocessing

    :param parent_conn: Parent connection of Pipe
    :param child_conn: Child connection of Pipe
    :param env: Gym environment we need multiprocessing for
    :type parent_conn: Multiprocessing Pipe Connection
    :type child_conn: Multiprocessing Pipe Connection
    :type env: Gym Environment
    """
    parent_conn.close()
    while True:
        cmd, data = child_conn.recv()
        if cmd == "step":
            observation, reward, done, info = env.step(data)
            child_conn.send((observation, reward, done, info))
        elif cmd == "seed":
            child_conn.send(env.seed(data))
        elif cmd == "reset":
            observation = env.reset()
            child_conn.send(observation)
        elif cmd == "render":
            child_conn.send(env.render())
        elif cmd == "close":
            env.close()
            child_conn.close()
            break
        elif cmd == "get_spaces":
            child_conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class VecEnv(ABC):
    """
    Base class for multiple environments.

    :param env: Gym environment to be vectorised
    :param n_envs: Number of environments
    :type env: Gym Environment
    :type n_envs: int
    """

    def __init__(self, envs: List, n_envs: int = 2):
        self.envs = envs
        self.env = envs[0]
        self._n_envs = n_envs
        self.episode_reward = torch.zeros(self.n_envs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def __getattr__(self, name: str) -> Any:
        env = super(VecEnv, self).__getattribute__("env")
        return getattr(env, name)

    def __iter__(self) -> Iterator:
        """
        Iterator object to iterate through each environment in vector
        """
        return (env for env in self.envs)

    def sample(self) -> List:
        """
        Return samples of actions from each environment
        """
        return torch.as_tensor([env.action_space.sample() for env in self.envs])

    def __getitem__(self, index: int) -> gym.Env:
        """
        Return environment at the given index

        :param index: Index at which the environment is
        :type index: int
        :returns: Gym Environment at given index of Vectorized Environment
        """
        return self.envs[index]

    def seed(self, seed: int):
        """
        Set seed for reproducibility in all environments
        """
        [env.seed(seed + idx) for idx, env in enumerate(self.envs)]

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def observation_spaces(self):
        return [i.observation_space for i in self.envs]

    @property
    def action_spaces(self):
        return [i.action_space for i in self.envs]

    @property
    def obs_shape(self):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_shape = self.observation_space.shape
        else:
            raise NotImplementedError
        return obs_shape

    @property
    def action_shape(self):
        if isinstance(self.action_space, gym.spaces.Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (1,)
        else:
            raise NotImplementedError
        return action_shape


class SerialVecEnv(VecEnv):
    """
    Constructs a wrapper for serial execution through envs.
    """

    def __init__(self, *args, **kwargs):
        super(SerialVecEnv, self).__init__(*args, **kwargs)
        self.states = torch.zeros(
            self.n_envs,
            *self.obs_shape,
        )
        self.rewards = torch.zeros(self.n_envs)
        self.dones = torch.zeros(self.n_envs)
        self.infos = [{} for _ in range(self.n_envs)]

    def step(self, actions: torch.Tensor) -> Tuple:
        """
        Steps through all envs serially

        :param actions: Actions from the model
        :type actions: Iterable of ints/floats
        """
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[i])
            self.states[i] = obs
            self.episode_reward[i] += reward
            self.rewards[i] = reward
            self.dones[i] = done
            self.infos[i] = info
        return (
            self.states.detach().clone(),
            self.rewards.detach().clone(),
            self.dones.detach().clone(),
            deepcopy(self.infos),
        )

    def reset(self) -> torch.Tensor:
        """
        Resets all envs
        """
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()
        self.episode_reward = torch.zeros(self.n_envs)

        return self.states.detach().clone()

    def reset_single_env(self, i: int) -> torch.Tensor:
        """
        Resets single environment
        """
        self.states[i] = self.envs[i].reset()
        self.episode_reward[i] = 0

        return self.states.detach().clone()

    def close(self):
        """
        Closes all envs
        """
        for env in self.envs:
            env.close()

    def get_spaces(self):
        return self.observation_space, self.action_space

    def images(self) -> List:
        """
        Returns an array of images from each env render
        """
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode="human"):
        """
        Renders all envs in a tiles format similar to baselines

        :param mode: (Can either be 'human' or 'rgb_array'. Displays tiled
            images in 'human' and returns tiled images in 'rgb_array')
        :type mode: string
        """
        self.env.render()


class SubProcessVecEnv(VecEnv):
    """
    Constructs a wrapper for parallel execution through envs.
    """

    def __init__(self, *args, **kwargs):
        super(SubProcessVecEnv, self).__init__(*args, **kwargs)

        self.procs = []
        self.parent_conns, self.child_conns = zip(
            *[mp.Pipe() for i in range(self._n_envs)]
        )

        for parent_conn, child_conn, env_fn in zip(
            self.parent_conns, self.child_conns, self.envs
        ):
            args = (parent_conn, child_conn, env_fn)
            process = mp.Process(target=worker, args=args, daemon=True)
            process.start()
            self.procs.append(process)
            child_conn.close()

    def get_spaces(self) -> Tuple:
        """
        Returns state and action spaces of environments
        """
        self.parent_conns[0].send(("get_spaces", None))
        observation_space, action_space = self.parent_conns[0].recv()
        return (observation_space, action_space)

    def seed(self, seed: int = None):
        """
        Sets seed for reproducability
        """
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("seed", seed + idx))

        return [parent_conn.recv() for parent_conn in self.parent_conns]

    def reset(self) -> torch.Tensor:
        """
        Resets environments

        :returns: States after environment reset
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        self.episode_reward = torch.zeros(self.n_envs)

        obs = [parent_conn.recv() for parent_conn in self.parent_conns]
        return torch.stack(obs)

    def step(self, actions: torch.Tensor) -> Tuple:
        """
        Steps through environments serially

        :param actions: Actions from the model
        :type actions: Iterable of ints/floats
        """
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(("step", action))
        self.waiting = True

        result = []
        for parent_conn in self.parent_conns:
            result.append(parent_conn.recv())
        self.waiting = False

        observations, rewards, dones, infos = zip(*result)
        self.episode_reward += torch.Tensor(rewards)
        return (torch.Tensor(v) for v in [observations, rewards, dones, infos])

    def close(self):
        """
        Closes all environments and processes
        """
        if self.waiting:
            for parent_conn in self.parent_conns:
                parent_conn.recv()
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
        for proc in self.procs:
            proc.join()
