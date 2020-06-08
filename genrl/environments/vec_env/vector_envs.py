from abc import ABC, abstractmethod

import multiprocessing as mp
import copy
import numpy as np


def worker(parent_conn, child_conn, env):
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

    def __init__(self, envs, n_envs=2):
        self.envs = envs
        self.single_env = envs[0]
        self._n_envs = n_envs

    def __getattr__(self, name):
        single_env = super(VecEnv, self).__getattribute__('single_env')
        return getattr(single_env, name)

    def __iter__(self):
        """
        Iterator object to iterate through each environment in vector
        """
        return (env for env in self.envs)

    def sample(self):
        """
        Return samples of actions from each environment
        """
        return [env.action_space.sample() for env in self.envs]

    def __getitem__(self, index):
        """
        Return environment at the given index
        """
        return self.envs[index]

    def seed(self, seed):
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
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def observation_spaces(self):
        return [i.observation_space for i in self.envs]

    @property
    def action_spaces(self):
        return [i.action_space for i in self.envs]


class SerialVecEnv(VecEnv):
    """
    Constructs a wrapper for serial execution through envs.
    """

    def __init__(self, *args, **kwargs):
        super(SerialVecEnv, self).__init__(*args, **kwargs)
        self.states = np.zeros(
            (self.n_envs, *self.observation_space.shape),
            dtype=self.observation_space.dtype,
        )
        self.rewards = np.zeros((self.n_envs))
        self.dones = np.zeros((self.n_envs))
        self.infos = [{} for _ in range(self.n_envs)]

    def step(self, actions):
        """
        Steps through all envs serially

        :param actions: Actions from the model
        :type actions: Iterable of ints/floats
        """
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(actions[i])
            if done:
                obs = env.reset()
            self.states[i] = obs
            self.rewards[i] = reward
            self.dones[i] = done
            self.infos[i] = info

        return (
            np.copy(self.states),
            self.rewards.copy(),
            self.dones.copy(),
            copy.deepcopy(self.infos),
        )

    def reset(self):
        """
        Resets all envs
        """
        for i, env in enumerate(self.envs):
            self.states[i] = env.reset()

        return np.copy(self.states)

    def close(self):
        """
        Closes all envs
        """
        for env in self.envs:
            env.close()

    def images(self):
        """
        Returns an array of images from each env render
        """
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode="human"):
        """
        Renders all envs in a tiles format similar to baselines

        :param mode: Can either be 'human' or 'rgb_array'. Displays tiled \
images in 'human' and returns tiled images in 'rgb_array'
        :type mode: string
        """
        self.envs[0].render()

        # Does not work need to debug how can vectorized envs be rendered
        # images = np.asarray(self.images())
        # N, H, W, C = images.shape
        # newW, newH = int(np.ceil(np.sqrt(W))), int(np.ceil(np.sqrt(H)))
        # images = np.array(list(images) + [images[0] * 0 for _ in range(N, newH * newW)])
        # out_image = images.reshape(newH, newW, H, W, C)
        # out_image = out_image.transpose(0, 2, 1, 3, 4)
        # out_image = out_image.reshape(newH * H, newW * W, C)
        # if mode == "human":
        #     # make_grid(self.images())
        #     import cv2  # noqa

        #     cv2.imshow("vecenv", out_image[:, :, ::-1])
        #     cv2.waitKey(1)
        # elif mode == "rgb_array":
        #     return out_image
        # else:
        #     raise NotImplementedError


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

    def get_spaces(self):
        """
        Returns state and action spaces of environments
        """
        self.parent_conns[0].send(("get_spaces", None))
        observation_space, action_space = self.parent_conns[0].recv()
        return (observation_space, action_space)

    def seed(self, seed=None):
        """
        Sets seed for reproducability
        """
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("seed", seed + idx))

        return [parent_conn.recv() for parent_conn in self.parent_conns]

    def reset(self):
        """
        Resets environments

        :returns: States after environment reset
        """
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        obs = [parent_conn.recv() for parent_conn in self.parent_conns]
        return obs

    def step(self, actions):
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
        return observations, rewards, dones, infos

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
