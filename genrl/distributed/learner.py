from genrl.distributed import Master, Node
from genrl.distributed.utils import remote_method, set_environ

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


class LearnerNode(Node):
    def __init__(self, name, master, parameter_server, experience_server, trainer, rank=None):
        super(LearnerNode, self).__init__(name, master, rank)
        self.parameter_server = parameter_server
        self.experience_server = experience_server

        mp.Process(
            target=self.run_paramater_server,
            args=(
                name,
                master.rref,
                master.address,
                master.port,
                master.world_size,
                self.rank,
                parameter_server.rref,
                experience_server.rref,
                trainer,
            ),
        )

    @staticmethod
    def train(
        name,
        master_rref,
        master_address,
        master_port,
        world_size,
        rank,
        parameter_server_rref,
        experience_server_rref,
        agent,
        trainer,
    ):
        print("Starting Learner")
        set_environ(master_address, master_port)
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        remote_method(Master.store_rref, master_rref, rpc.RRef(trainer), name)
        trainer.train_wrapper(parameter_server_rref, experience_server_rref)
        rpc.shutdown()
        print("Shutdown learner")
