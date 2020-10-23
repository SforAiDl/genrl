from genrl.distributed import Node
from genrl.distributed.core import get_proxy, store_rref

import torch.distributed.rpc as rpc


class LearnerNode(Node):
    def __init__(
        self,
        name,
        master,
        parameter_server_name,
        experience_server_name,
        trainer,
        rank=None,
    ):
        super(LearnerNode, self).__init__(name, master, rank)
        self.init_proc(
            target=self.learn,
            kwargs=dict(
                parameter_server_name=parameter_server_name,
                experience_server_name=experience_server_name,
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
        rpc_backend,
        **kwargs,
    ):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank, backend=rpc_backend)
        print(f"{name}: Initialised RPC")
        store_rref(name, rpc.RRef(trainer))
        parameter_server = get_proxy(parameter_server_name)
        experience_server = get_proxy(experience_server_name)
        print(f"{name}: Beginning training")
        trainer.train_wrapper(parameter_server, experience_server)
        rpc.shutdown()
