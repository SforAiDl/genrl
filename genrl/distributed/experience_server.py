from genrl.distributed import Node
from genrl.distributed.core import store_rref

import torch.distributed.rpc as rpc


class ExperienceServer(Node):
    def __init__(self, name, master, buffer, rank=None):
        super(ExperienceServer, self).__init__(name, master, rank)
        self.init_proc(
            target=self.run_paramater_server,
            kwargs=dict(buffer=buffer),
        )
        self.start_proc()

    @staticmethod
    def run_paramater_server(name, world_size, rank, buffer, **kwargs):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        print("inited exp rpcs")
        rref = rpc.RRef(buffer)
        print(rref)
        store_rref(name, rref)
        print("serving buffer")
        rpc.shutdown()
