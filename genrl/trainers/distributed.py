import copy
import multiprocessing as mp
import threading
from typing import Type, Union

import gym
import numpy as np
import reverb
import tensorflow as tf
import torch

from genrl.trainers import Trainer


class DistributedOffPolicyTrainer:
    """Distributed Off Policy Trainer Class

    Trainer class for Distributed Off Policy Agents

    """

    def __init__(
        self,
        agent,
        env,
        buffer_server_port=None,
        param_server_port=None,
        **kwargs,
    ):
        self.env = env
        self.agent = agent
        self.buffer_server_port = buffer_server_port
        self.param_server_port = param_server_port

    def train(
        self, n_actors, max_buffer_size, batch_size, max_updates, update_interval
    ):
        buffer_server = reverb.Server(
            tables=[
                reverb.Table(
                    name="replay_buffer",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=max_buffer_size,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                )
            ],
            port=self.buffer_server_port,
        )
        buffer_server_address = f"localhost:{buffer_server.port}"

        param_server = reverb.Server(
            tables=[
                reverb.Table(
                    name="param_buffer",
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=1,
                    rate_limiter=reverb.rate_limiters.MinSize(1),
                )
            ],
            port=self.param_server_port,
        )
        param_server_address = f"localhost:{param_server.port}"

        actor_procs = []
        for _ in range(n_actors):
            p = threading.Thread(
                target=run_actor,
                args=(
                    copy.deepcopy(self.agent),
                    copy.deepcopy(self.env),
                    buffer_server_address,
                    param_server_address,
                ),
                daemon=True,
            )
            p.start()
            actor_procs.append(p)

        learner_proc = threading.Thread(
            target=run_learner,
            args=(
                copy.deepcopy(self.agent),
                max_updates,
                update_interval,
                buffer_server_address,
                param_server_address,
                batch_size,
            ),
            daemon=True,
        )
        learner_proc.daemon = True
        learner_proc.start()
        learner_proc.join()

        # param_client = reverb.Client(param_server_address)
        # self.agent.replay_buffer = ReverbReplayDataset(
        #     self.agent.env, buffer_server_address, batch_size
        # )

        # for _ in range(max_updates):
        #     self.agent.update_params(update_interval)
        #     params = self.agent.get_weights()
        #     param_client.insert(params.values(), {"param_buffer": 1})
        #     print("weights updated")
        #     # print(list(param_client.sample("param_buffer")))


def run_actor(agent, env, buffer_server_address, param_server_address):
    buffer_client = reverb.Client(buffer_server_address)
    param_client = reverb.TFClient(param_server_address)

    state = env.reset().astype(np.float32)

    for i in range(10):
        # params = param_client.sample("param_buffer", [])
        # print("Sampling done")
        # print(list(params))
        # agent.load_weights(params)

        action = agent.select_action(state).numpy()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.astype(np.float32)
        reward = np.array([reward]).astype(np.float32)
        done = np.array([done]).astype(np.bool)

        buffer_client.insert([state, action, reward, next_state, done], {"replay_buffer": 1})
        print("transition inserted")
        state = env.reset().astype(np.float32) if done else next_state.copy()


def run_learner(
    agent,
    max_updates,
    update_interval,
    buffer_server_address,
    param_server_address,
    batch_size,
):
    param_client = reverb.Client(param_server_address)
    agent.replay_buffer = ReverbReplayDataset(
        agent.env, buffer_server_address, batch_size
    )
    for _ in range(max_updates):
        agent.update_params(update_interval)
        params = agent.get_weights()
        param_client.insert(params.values(), {"param_buffer": 1})
        print("weights updated")
        # print(list(param_client.sample("param_buffer")))


class ReverbReplayDataset:
    def __init__(self, env, address, batch_size):
        action_dtype = (
            np.int64
            if isinstance(env.action_space, gym.spaces.discrete.Discrete)
            else np.float32
        )
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        reward_shape = 1
        done_shape = 1

        self._dataset = reverb.ReplayDataset(
            server_address=address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=2 * batch_size,
            dtypes=(np.float32, action_dtype, np.float32, np.float32, np.bool),
            shapes=(
                tf.TensorShape(obs_shape),
                tf.TensorShape(action_shape),
                tf.TensorShape(reward_shape),
                tf.TensorShape(obs_shape),
                tf.TensorShape(done_shape),
            ),
        )
        self._data_iter = self._dataset.batch(batch_size).as_numpy_iterator()

    def sample(self, *args, **kwargs):
        sample = next(self._data_iter)
        obs, a, r, next_obs, d = [torch.from_numpy(t).float() for t in sample.data]
        return obs, a, r, next_obs, d
