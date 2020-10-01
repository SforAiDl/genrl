from genrl.distributed import Master, Node
from genrl.distributed.utils import remote_method, set_environ

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


class ParameterServer(Node):
    def __init__(self, name, master, init_params, rank=None):
        super(ParameterServer, self).__init__(name, master, rank)
        mp.Process(
            target=self.run_paramater_server,
            args=(name, master.rref, master.address, master.port, master.world_size, self.rank, init_params),
        )

    @staticmethod
    def run_paramater_server(
        name, master_rref, master_address, master_port, world_size, rank, init_params
    ):
        print("Starting Parameter Server")
        set_environ(master_address, master_port)
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        params = init_params
        remote_method(Master.store_rref, master_rref, rpc.RRef(params), name)
        rpc.shutdown()
        print("Shutdown parameter server")


class WeightHolder:
    def __init__(self, init_weights):
        self._weights = init_weights

    def store_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights
