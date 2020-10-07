import torch.distributed.rpc as rpc

import threading

from abc import ABC, abstractmethod
import torch.multiprocessing as mp
import os
import time

_rref_reg = {}
_global_lock = threading.Lock()


def _get_rref(idx):
    global _rref_reg
    with _global_lock:
        if idx in _rref_reg.keys():
            return _rref_reg[idx]
        else:
            return None


def _store_rref(idx, rref):
    global _rref_reg
    with _global_lock:
        if idx in _rref_reg.keys():
            raise Warning(
                f"Re-assigning RRef for key: {idx}. Make sure you are not using duplicate names for nodes"
            )
        _rref_reg[idx] = rref


def get_rref(idx):
    rref = rpc.rpc_sync("master", _get_rref, args=(idx,))
    while rref is None:
        time.sleep(0.5)
        rref = rpc.rpc_sync("master", _get_rref, args=(idx,))
    return rref


def store_rref(idx, rref):
    rpc.rpc_sync("master", _store_rref, args=(idx, rref))


def set_environ(address, port):
    os.environ["MASTER_ADDR"] = str(address)
    os.environ["MASTER_PORT"] = str(port)


class Node:
    def __init__(self, name, master, rank):
        self._name = name
        self.master = master
        if rank is None:
            self._rank = master.node_count
        elif rank >= 0 and rank < master.world_size:
            self._rank = rank
        elif rank >= master.world_size:
            raise ValueError("Specified rank greater than allowed by world size")
        else:
            raise ValueError("Invalid value of rank")
        self.p = None

    def __del__(self):
        if self.p is None:
            raise RuntimeWarning(
                "Removing node when process was not initialised properly"
            )
        else:
            self.p.join()

    @staticmethod
    def _target_wrapper(target, **kwargs):
        pid = os.getpid()
        print(f"Starting {kwargs['name']} with pid {pid}")
        set_environ(kwargs["master_address"], kwargs["master_port"])
        target(**kwargs)
        print(f"Shutdown {kwargs['name']} with pid {pid}")

    def init_proc(self, target, kwargs):
        kwargs.update(
            dict(
                name=self.name,
                master_address=self.master.address,
                master_port=self.master.port,
                world_size=self.master.world_size,
                rank=self.rank,
            )
        )
        self.p = mp.Process(target=self._target_wrapper, args=(target,), kwargs=kwargs)

    def start_proc(self):
        if self.p is None:
            raise RuntimeError("Trying to start uninitialised process")
        self.p.start()

    @property
    def name(self):
        return self._name

    @property
    def rref(self):
        return get_rref(self.name)

    @property
    def rank(self):
        return self._rank


def _run_master(world_size):
    print(f"Starting master at {os.getpid()}")
    rpc.init_rpc("master", rank=0, world_size=world_size)
    rpc.shutdown()


class Master:
    def __init__(self, world_size, address="localhost", port=29501):
        set_environ(address, port)
        self._world_size = world_size
        self._address = address
        self._port = port
        self._node_counter = 0
        self.p = mp.Process(target=_run_master, args=(world_size,))
        self.p.start()

    def __del__(self):
        if self.p is None:
            raise RuntimeWarning(
                "Shutting down master when it was not initialised properly"
            )
        else:
            self.p.join()

    @property
    def world_size(self):
        return self._world_size

    @property
    def address(self):
        return self._address

    @property
    def port(self):
        return self._port

    @property
    def node_count(self):
        return self._node_counter
