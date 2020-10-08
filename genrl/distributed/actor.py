from genrl.distributed.core import Node
from genrl.distributed.core import get_rref, store_rref
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
        **kwargs,
    ):
        rpc.init_rpc(name=name, world_size=world_size, rank=rank)
        print(f"{name}: RPC Initialised")
        rref = rpc.RRef(agent)
        store_rref(name, rref)
        parameter_server_rref = get_rref(parameter_server_name)
        experience_server_rref = get_rref(experience_server_name)
        learner_rref = get_rref(learner_name)
        print(f"{name}: Begining experience collection")
        while not learner_rref.rpc_sync().is_done():
            agent.load_weights(parameter_server_rref.rpc_sync().get_weights())
            collect_experience(agent, experience_server_rref)

        rpc.shutdown()
