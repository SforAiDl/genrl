from genrl.distributed.core import (
    DistributedTrainer,
    Master,
    Node,
)
from genrl.distributed.parameter_server import WeightHolder
from genrl.distributed.utils import remote_method, set_environ

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


class ActorNode(Node):
    def __init__(
        self,
        name,
        master,
        parameter_server,
        experience_server,
        learner,
        agent,
        collect_experience,
        rank=None
    ):
        super(ActorNode, self).__init__(name, master, rank)
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
                learner.rref,
                agent,
                collect_experience,
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
        learner_rref,
        agent,
        collect_experience,

    ):
        print("Starting Actor")
        set_environ(master_address, master_port)
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        remote_method(Master.store_rref, master_rref, rpc.RRef(agent), name)
        while not remote_method(DistributedTrainer.is_done(), learner_rref):
            agent.load_weights(
                remote_method(WeightHolder.get_weights(), parameter_server_rref)
            )
            print("Done loadiing weights")
            collect_experience(agent, experience_server_rref)
            print("Done collecting experience")

        rpc.shutdown()
        print("Shutdown actor")
