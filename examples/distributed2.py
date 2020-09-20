from genrl.core.buffers import ReplayBuffer
import os

from genrl.agents import DDPG
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse

import copy

import gym
import numpy as np

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# to call a function on an rref, we could do the following
# _remote_method(some_func, rref, *args)

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


gloabl_lock = mp.Lock()


class ParamServer:
    def __init__(self, init_params):
        self.params = init_params
        # self.lock = mp.Lock()

    def store_params(self, new_params):
        # with self.lock:
        with gloabl_lock:
            self.params = new_params

    def get_params(self):
        # with self.lock:
        with gloabl_lock:
            return self.params


class DistributedReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.len = 0
        self._buffer = ReplayBuffer(self.size)


class DistributedOffPolicyTrainer:
    """Distributed Off Policy Trainer Class

    Trainer class for Distributed Off Policy Agents

    """
    def __init__(
        self,
        agent,
        env,
        **kwargs,
    ):
        self.env = env
        self.agent = agent

    def train(
        self, n_actors, max_buffer_size, batch_size, max_updates, update_interval
    ):

        print("a")
        world_size = n_actors + 2
        completed = mp.Value("i", 0)
        print("a")
        param_server_rref_q = mp.Queue(1)
        param_server_p = mp.Process(
            target=run_param_server, args=(param_server_rref_q, world_size,)
        )
        param_server_p.start()
        param_server_rref = param_server_rref_q.get()
        param_server_rref_q.close()

        print("a")
        buffer_rref_q = mp.Queue(1)
        buffer_p = mp.Process(target=run_buffer, args=(max_buffer_size, buffer_rref_q, world_size,))
        buffer_p.start()
        buffer_rref = buffer_rref_q.get()
        buffer_rref_q.close()
        print("a")

        actor_ps = []
        for i in range(n_actors):
            a_p = mp.Process(
                target=run_actor,
                args=(
                    i,
                    copy.deepcopy(self.agent),
                    copy.deepcopy(self.env),
                    param_server_rref,~
                    buffer_rref,
                    world_size,
                    completed
                ),
            )
            a_p.start()
            actor_ps.append(a_p)

        learner_p = mp.Process(
            target=run_learner,
            args=(max_updates, batch_size, self.agent, param_server_rref, buffer_rref, world_size, completed),
        )
        learner_p.start()

        learner_p.join()
        for a in actor_ps:
            a.join()
        buffer_p.join()
        param_server_p.join()


def run_param_server(q, world_size):
    print("Running parameter server")
    rpc.init_rpc(name="param_server", rank=0, world_size=world_size)
    print("d")
    param_server = ParamServer(None)
    param_server_rref = rpc.RRef(param_server)
    q.put(param_server_rref)
    rpc.shutdown()
    print("param server shutting down")


def run_buffer(max_buffer_size, q, world_size):
    print("Running buffer server")
    rpc.init_rpc(name="buffer", rank=1, world_size=world_size)
    buffer = ReplayBuffer(max_buffer_size)
    buffer_rref = rpc.RRef(buffer)
    q.put(buffer_rref)
    rpc.shutdown()
    print("buffer shutting down")


def run_learner(max_updates, batch_size, agent, param_server_rref, buffer_rref, world_size, completed):
    print("Running learner")
    rpc.init_rpc(name="learner", rank=world_size - 1, world_size=world_size)
    i = 0
    while i < max_updates:
        batch = _remote_method(ReplayBuffer.sample, buffer_rref, batch_size)
        if batch is None:
            continue
        agent.update_params(batch)
        _remote_method(ParamServer.store_params, param_server_rref, agent.get_weights())
        print("weights updated")
        i += 1
        print(i)
    completed.value = 1
    rpc.shutdown()
    print("learner shutting down")


def run_actor(i, agent, env, param_server_rref, buffer_rref, world_size, completed):
    print(f"Running actor {i}")

    rpc.init_rpc(name=f"action_{i}", rank=i + 1, world_size=world_size)

    state = env.reset().astype(np.float32)

    while not completed.value == 1:
        params = _remote_method(ParamServer.get_params, param_server_rref)
        agent.load_weights(params)

        action = agent.select_action(state).numpy()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.astype(np.float32)
        reward = np.array([reward]).astype(np.float32)
        done = np.array([done]).astype(np.bool)

        print("attempting to insert transition")
        _remote_method(ReplayBuffer.push, buffer_rref, [state, action, reward, next_state, done])
        print("inserted transition")
        state = env.reset().astype(np.float32) if done else next_state.copy()

    rpc.shutdown()
    print("actor shutting down")

env = gym.make("Pendulum-v0")
agent = DDPG("mlp", env)

trainer = DistributedOffPolicyTrainer(agent, env)
trainer.train(
    n_actors=1, max_buffer_size=100, batch_size=1, max_updates=100, update_interval=1
)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Parameter-Server RPC based training")
#     parser.add_argument(
#         "--world_size",
#         type=int,
#         default=4,
#         help="""Total number of participating processes. Should be the number
#         of actors + 3.""")
#     parser.add_argument(
#         "--run",
#         type=str,
#         default="param_server",
#         choices=["param_server", "buffer", "learner", "actor"],
#         help="Which program to run")
#     parser.add_argument(
#         "--master_addr",
#         type=str,
#         default="localhost",
#         help="""Address of master, will default to localhost if not provided.
#         Master must be able to accept network traffic on the address + port.""")
#     parser.add_argument(
#         "--master_port",
#         type=str,
#         default="29500",
#         help="""Port that master is listening on, will default to 29500 if not
#         provided. Master must be able to accept network traffic on the host and port.""")

#     args = parser.parse_args()

#     os.environ['MASTER_ADDR'] = args.master_addr
#     os.environ["MASTER_PORT"] = args.master_port

#     processes = []
#     world_size = args.world_size
#     if args.run == "param_server":
#         p = mp.Process(target=run_param_server, args=(world_size))
#         p.start()
#         processes.append(p)
#     elif args.run == "buffer":
#         p = mp.Process(target=run_buffer, args=(world_size))
#         p.start()
#         processes.append(p)
#         # Get data to train on
#         train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('../data', train=True, download=True,
#                         transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.1307,), (0.3081,))
#                         ])),
#             batch_size=32, shuffle=True,)
#         test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST(
#                 '../data',
#                 train=False,
#                 transform=transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.1307,), (0.3081,))
#                             ])),
#             batch_size=32,
#             shuffle=True,
#         )
#         # start training worker on this node
#         p = mp.Process(
#             target=run_worker,
#             args=(
#                 args.rank,
#                 world_size, args.num_gpus,
#                 train_loader,
#                 test_loader))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()
