from genrl.distributed import Master, Node
from genrl.distributed.utils import remote_method, set_environ

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from genrl.core import ReplayBuffer


class ExperienceServer(Node):
    def __init__(self, name, master, size, rank=None):
        super(ExperienceServer, self).__init__(name, master, rank)
        mp.Process(
            target=self.run_paramater_server,
            args=(name, master.rref, master.address, master.port, master.world_size, rank, size),
        )

    @staticmethod
    def run_paramater_server(name, master_rref, master_address, master_port, world_size, rank, size):
        print("Starting Parameter Server")
        set_environ(master_address, master_port)
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        buffer = ReplayBuffer(size)
        remote_method(Master.store_rref, master_rref, rpc.RRef(buffer), name)
        rpc.shutdown()
        print("Shutdown experience server")
