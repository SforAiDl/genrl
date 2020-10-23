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
    def run_paramater_server(name, world_size, rank, buffer, rpc_backend, **kwargs):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank, backend=rpc_backend)
        print(f"{name}: Initialised RPC")
        store_rref(name, rpc.RRef(buffer))
        print(f"{name}: Serving experience buffer")
        rpc.shutdown()
