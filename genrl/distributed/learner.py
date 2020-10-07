from genrl.distributed import Node
from genrl.distributed.core import get_rref, store_rref

import torch.distributed.rpc as rpc


class LearnerNode(Node):
    def __init__(
        self, name, master, parameter_server, experience_server, trainer, rank=None
    ):
        super(LearnerNode, self).__init__(name, master, rank)
        self.parameter_server = parameter_server
        self.experience_server = experience_server

        self.init_proc(
            target=self.learn,
            kwargs=dict(
                parameter_server_name=self.parameter_server.name,
                experience_server_name=self.experience_server.name,
                trainer=trainer,
            ),
        )
        self.start_proc()

    @staticmethod
    def learn(
        name,
        world_size,
        rank,
        parameter_server_name,
        experience_server_name,
        trainer,
        **kwargs,
    ):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        print(f"{name}: Initialised RPC")
        rref = rpc.RRef(trainer)
        store_rref(name, rref)
        parameter_server_rref = get_rref(parameter_server_name)
        experience_server_rref = get_rref(
            experience_server_name,
        )
        print(f"{name}: Beginning training")
        trainer.train_wrapper(parameter_server_rref, experience_server_rref)
        rpc.shutdown()
