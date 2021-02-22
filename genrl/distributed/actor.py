from genrl.distributed.core import Node
from genrl.distributed.core import get_proxy, store_rref
import torch.distributed.rpc as rpc


class ActorNode(Node):
    def __init__(
        self,
        name,
        master,
        parameter_server_name,
        experience_server_name,
        learner_name,
        agent,
        collect_experience,
        rank=None,
    ):
        super(ActorNode, self).__init__(name, master, rank)
        self.init_proc(
            target=self.act,
            kwargs=dict(
                parameter_server_name=parameter_server_name,
                experience_server_name=experience_server_name,
                learner_name=learner_name,
                agent=agent,
                collect_experience=collect_experience,
            ),
        )
        self.start_proc()

    @staticmethod
    def act(
        name,
        world_size,
        rank,
        parameter_server_name,
        experience_server_name,
        learner_name,
        agent,
        collect_experience,
        rpc_backend,
        **kwargs,
    ):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank, backend=rpc_backend)
        print(f"{name}: RPC Initialised")
        store_rref(name, rpc.RRef(agent))
        parameter_server = get_proxy(parameter_server_name)
        experience_server = get_proxy(experience_server_name)
        learner = get_proxy(learner_name)
        print(f"{name}: Begining experience collection")
        collect_experience(agent, parameter_server, experience_server, learner)
        rpc.shutdown()
