from genrl.distributed import Node
from genrl.distributed.core import store_rref

import torch.distributed.rpc as rpc


class ParameterServer(Node):
    def __init__(self, name, master, init_params, rank=None):
        super(ParameterServer, self).__init__(name, master, rank)
        self.init_proc(
            target=self.run_paramater_server,
            kwargs=dict(init_params=init_params),
        )
        self.start_proc()

    @staticmethod
    def run_paramater_server(name, world_size, rank, init_params, **kwargs):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        print(f"{name}: Initialised RPC")
        params = WeightHolder(init_weights=init_params)
        store_rref(name, rpc.RRef(params))
        print(f"{name}: Serving parameters")
        rpc.shutdown()


class WeightHolder:
    def __init__(self, init_weights):
        self._weights = init_weights

    def store_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights
