import torch.distributed.rpc as rpc

from genrl.distributed.utils import remote_method, set_environ


class Node:
    def __init__(self, name, master, rank):
        self._name = name
        self.master = master
        self.master.increment_node_count()
        if rank is None:
            self._rank = master.node_count
        elif rank > 0 and rank < master.world_size:
            self._rank = rank
        elif rank == 0:
            raise ValueError("Rank of 0 is invalid for node")
        elif rank >= master.world_size:
            raise ValueError("Specified rank greater than allowed by world size")
        else:
            raise ValueError("Invalid value of rank")

    @property
    def rref(self):
        return self.master[self.name]

    @property
    def rank(self):
        return self._rank


class Master(Node):
    def __init__(self, world_size, address="localhost", port=29500):
        print("initing master")
        set_environ(address, port)
        print("initing master")
        rpc.init_rpc(name="master", rank=0, world_size=world_size)
        print("initing master")
        self._world_size = world_size
        self._rref = rpc.RRef(self)
        print("initing master")
        self._rref_reg = {}
        self._address = address
        self._port = port
        self._node_counter = 0

    def store_rref(self, parent_rref, idx):
        self._rref_reg[idx] = parent_rref

    def fetch_rref(self, idx):
        return self._rref_reg[idx]

    @property
    def rref(self):
        return self._rref

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

    def increment_node_counter(self):
        self._node_counter += 1
        if self.node_count >= self.world_size:
            raise Exception("Attempt made to add more nodes than specified by world size")


class DistributedTrainer:
    def __init__(self, agent):
        self.agent = agent
        self._completed_training_flag = False

    def train(self, parameter_server_rref, experience_server_rref):
        raise NotImplementedError

    def train_wrapper(self, parameter_server_rref, experience_server_rref):
        self._completed_training_flag = False
        self.train(parameter_server_rref, experience_server_rref)
        self._completed_training_flag = True

    def is_done(self):
        return self._completed_training_flag
