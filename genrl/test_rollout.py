from genrl.deep.common.rollout_storage import RolloutBuffer, BaseBuffer
import reverb
import torch
import numpy as np
import tensorflow as tf

from genrl.environments import VectorEnv


class RolloutBuffer1(BaseBuffer):
    def __init__(
        self,
        size,
        batch_size,
        obs_shape,
        action_shape,
        action_dtype="discrete",
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
        self.action_dtype = np.int64 if action_dtype == "discrete" else np.float32

        self._pos = 0
        self._queue = reverb.Table.queue(name="queue", max_size=self.size)
        self._server = reverb.Server(tables=[self._queue], port=None)
        self._server_address = f"localhost:{self._server.port}"
        self._client = reverb.Client(self._server_address)
        self._dataset = reverb.ReplayDataset(
            server_address=self._server_address,
            table="queue",
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

        self._client.insert(i, priorities={"queue": 1.0})
        if self._pos < self.size:
            self._pos += 1

    def extend(self, inp):
        for sample in inp:
            self.push(sample)

    def get(self):
        sample = next(self._iterator)
        obs, a, r, next_obs, d = [torch.from_numpy(t) for t in sample.data]
        return obs, a, r, next_obs, d

    def __len__(self):
        return self._pos

    def __del__(self):
        self._server.stop()


class ReplayBuffer2:
    def __init__(
        self,
        size,
        batch_size,
        obs_shape,
        action_shape,
        action_dtype="discrete",
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
        self.action_dtype = np.int64 if action_dtype == "discrete" else np.float32
        self.batch_size = batch_size
        self._pos = 0
        self._table = reverb.Table(
            name="replay_buffer",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=size,
            rate_limiter=reverb.rate_limiters.MinSize(2),
            # signature=
        )
        self._server = reverb.Server(tables=[self._table], port=None)
        self._server_address = f"localhost:{self._server.port}"
        self._client = reverb.Client(self._server_address)

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

    def sample(self):
        samples = self._client.sample(
            table="replay_buffer", num_samples=self.batch_size
        )
        batch = [[], [], [], [], []]
        for s in samples:
            for i in range(5):
                batch[i].append(torch.from_numpy(s[0].data[i]))
        obs, a, r, next_obs, d = (torch.stack(t) for t in batch)
        return obs, a, r, next_obs, d

    def __len__(self):
        return self._pos

    def __del__(self):
        self._server.stop()


env = env = VectorEnv("CartPole-v1", 1)
obs = env.reset()
a = env.action_space.sample()
next_obs, r, d, _ = env.step([a])
r = np.array([r], dtype=np.float32)
d = np.array([d], dtype=np.bool)

size = 5000
batch_size = 1024
rb = ReplayBuffer(size=size, obs_shape=(4,), action_shape=(1,))
rb1 = ReplayBuffer1(size=size, batch_size=batch_size, obs_shape=(4,), action_shape=(1,))
rb2 = ReplayBuffer2(size=size, batch_size=batch_size, obs_shape=(4,), action_shape=(1,))

import argparse

parser = argparse.ArgumentParser(description="Test performance of buffers")
parser.add_argument(
    "-n", metavar="N", type=int, default=1000, help="Number of iterations to time for"
)
args = parser.parse_args()


import timeit

print(f"\nBenchmarking for {args.n} executions ...\n")

rb_push = (
    timeit.timeit(
        "rb.push((obs, a, r.reshape(1), next_obs, d.reshape(1)))", number=args.n, globals=globals()
    )
    / args.n
)
rb_sample = timeit.timeit("rb.sample(batch_size)", number=args.n, globals=globals()) / args.n
del rb

rb1_push = (
    timeit.timeit(
        "rb1.push((obs, a, r, next_obs, d))", number=args.n, globals=globals()
    )
    / args.n
)
rb1_sample = timeit.timeit("rb1.sample()", number=args.n, globals=globals()) / args.n
del rb1

rb2_push = (
    timeit.timeit(
        "rb2.push((obs, a, r, next_obs, d))", number=args.n, globals=globals()
    )
    / args.n
)
rb2_sample = timeit.timeit("rb2.sample()", number=args.n, globals=globals()) / args.n
del rb2

print()
print(f"Existing ReplayBuffer: Push = {rb_push} | Sample = {rb_sample}")
print(f"With td dataset: Push = {rb1_push} | Sample = {rb1_sample}")
print(f"With regular reverb sampling: Push = {rb2_push} | Sample = {rb2_sample}")
