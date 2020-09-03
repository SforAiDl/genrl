import copy
import multiprocessing as mp
from typing import Type, Union

import numpy as np
import reverb
import tensorflow as tf
import torch

from genrl.trainers import Trainer


class ReverbReplayBuffer:
    def __init__(
        self,
        size,
        batch_size,
        obs_shape,
        action_shape,
        discrete=True,
        reward_shape=(1,),
        done_shape=(1,),
        n_envs=1,
    ):
        self.size = size
        self.obs_shape = (n_envs, *obs_shape)
        self.action_shape = (n_envs, *action_shape)
        self.reward_shape = (n_envs, *reward_shape)
        self.done_shape = (n_envs, *done_shape)
        self.n_envs = n_envs
        self.action_dtype = np.int64 if discrete else np.float32

        self._pos = 0
        self._table = reverb.Table(
            name="replay_buffer",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self.size,
            rate_limiter=reverb.rate_limiters.MinSize(2),
        )
        self._server = reverb.Server(tables=[self._table], port=None)
        self._server_address = f"localhost:{self._server.port}"
        self._client = reverb.Client(self._server_address)
        self._dataset = reverb.ReplayDataset(
            server_address=self._server_address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(np.float32, self.action_dtype, np.float32, np.float32, np.bool),
            shapes=(
                tf.TensorShape([n_envs, *obs_shape]),
                tf.TensorShape([n_envs, *action_shape]),
                tf.TensorShape([n_envs, *reward_shape]),
                tf.TensorShape([n_envs, *obs_shape]),
                tf.TensorShape([n_envs, *done_shape]),
            ),
        )
        self._iterator = self._dataset.batch(batch_size).as_numpy_iterator()

    def push(self, inp):
        i = []
        i.append(np.array(inp[0], dtype=np.float32).reshape(self.obs_shape))
        i.append(np.array(inp[1], dtype=self.action_dtype).reshape(self.action_shape))
        i.append(np.array(inp[2], dtype=np.float32).reshape(self.reward_shape))
        i.append(np.array(inp[3], dtype=np.float32).reshape(self.obs_shape))
        i.append(np.array(inp[4], dtype=np.bool).reshape(self.done_shape))

        self._client.insert(i, priorities={"replay_buffer": 1.0})
        if self._pos < self.size:
            self._pos += 1

    def extend(self, inp):
        for sample in inp:
            self.push(sample)

    def sample(self, *args, **kwargs):
        sample = next(self._iterator)
        obs, a, r, next_obs, d = [torch.from_numpy(t).float() for t in sample.data]
        return obs, a, r.reshape(-1, self.n_envs), next_obs, d.reshape(-1, self.n_envs)

    def __len__(self):
        return self._pos

    def __del__(self):
        self._server.stop()


class DistributedOffPolicyTrainer(Trainer):
    """Distributed Off Policy Trainer Class

    Trainer class for Distributed Off Policy Agents

    """

    def __init__(
        self,
        *args,
        env,
        agent,
        max_ep_len: int = 500,
        max_timesteps: int = 5000,
        update_interval: int = 50,
        buffer_server_port=None,
        param_server_port=None,
        **kwargs,
    ):
        super(DistributedOffPolicyTrainer, self).__init__(
            *args, off_policy=True, max_timesteps=max_timesteps, **kwargs
        )
        self.env = env
        self.agent = agent
        self.max_ep_len = max_ep_len
        self.update_interval = update_interval
        self.buffer_server_port = buffer_server_port
        self.param_server_port = param_server_port

    def train(self, n_actors, max_buffer_size, batch_size, max_updates):
        buffer_server = reverb.Server(
            tables=[
                reverb.Table(
                    name="replay_buffer",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=max_buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(2),
                )
            ],
            port=self.buffer_server_port,
        )
        buffer_server_address = f"localhost:{self.buffer_server.port}"

        param_server = reverb.Server(
            tables=[
                reverb.Table(
                    name="replay_buffer",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=1,
                )
            ],
            port=self.param_server_port,
        )
        param_server_address = f"localhost:{self.param_server.port}"

        actor_procs = []
        for _ in range(n_actors):
            p = mp.Process(
                target=self._run_actor,
                args=(
                    copy.deepcopy(self.agent),
                    copy.deepcopy(self.env),
                    buffer_server_address,
                    param_server_address,
                ),
            )
            p.daemon = True
            actor_procs.append(p)

        learner_proc = mp.Process(
            target=self._run_learner,
            args=(
                self.agent,
                max_updates,
                buffer_server_address,
                param_server_address,
                batch_size,
            ),
        )
        learner_proc.daemon = True

    def _run_actor(self, agent, env, buffer_server_address, param_server_address):
        buffer_client = reverb.Client(buffer_server_address)
        param_client = reverb.Client(param_server_address)

        state = env.reset()

        while True:
            params = param_client.sample(table="replay_buffer")
            agent.load_weights(params)

            action = self.get_action(state)
            next_state, reward, done, info = self.env.step(action)

            state = next_state.clone()

            buffer_client.insert([state, action, reward, done, next_state])

    def _run_learner(
        self,
        agent,
        max_updates,
        buffer_server_address,
        param_server_address,
        batch_size,
    ):
        param_client = reverb.Client(param_server_address)
        dataset = reverb.ReplayDataset(
            server_address=buffer_server_address,
            table="replay_buffer",
        )
        data_iter = dataset.batch(batch_size).as_numpy_iterator()

        for _ in range(max_updates):
            sample = next(data_iter)
            agent.update_params(sample)
            param_client.insert(agent.get_weights())
