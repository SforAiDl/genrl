from genrl.distributed import (
    Master,
    ExperienceServer,
    ParameterServer,
    ActorNode,
    LearnerNode,
    remote_method,
    DistributedTrainer,
    WeightHolder,
)
from genrl.core import ReplayBuffer
from genrl.agents import DDPG
import gym


N_ACTORS = 2
BUFFER_SIZE = 10
MAX_ENV_STEPS = 100
TRAIN_STEPS = 10
BATCH_SIZE = 5


def collect_experience(agent, experience_server_rref):
    obs = agent.env.reset()
    done = False
    for i in range(MAX_ENV_STEPS):
        action = agent.select_action(obs)
        next_obs, reward, done, info = agent.env.step()
        print("Sending experience")
        remote_method(ReplayBuffer.push, experience_server_rref)
        print("Done sending experience")
        if done:
            break


class MyTrainer(DistributedTrainer):
    def __init__(self, agent, train_steps, batch_size):
        super(MyTrainer, self).__init__(agent)
        self.train_steps = train_steps
        self.batch_size = batch_size

    def train(self, parameter_server_rref, experience_server_rref):
        for i in range(self.train_steps):
            batch = remote_method(
                ReplayBuffer.sample, parameter_server_rref, self.batch_size
            )
            if batch is None:
                continue
            self.agent.update_params(batch)
            print("Storing weights")
            remote_method(
                WeightHolder.store_weights,
                parameter_server_rref,
                self.agent.get_weights(),
            )
            print("Done storing weights")


master = Master(world_size=6, address="localhost")
print("inited master")
env = gym.make("Pendulum-v0")
agent = DDPG(env)
parameter_server = ParameterServer("param-0", master, agent.get_weights, rank=1)
experience_server = ExperienceServer("experience-0", master, BUFFER_SIZE, rank=2)
trainer = MyTrainer(agent, TRAIN_STEPS, BATCH_SIZE)
learner = LearnerNode("learner-0", master, parameter_server, experience_server, trainer, rank=3)
actors = [
    ActorNode(
        f"actor-{i}",
        master,
        parameter_server,
        experience_server,
        learner,
        agent,
        collect_experience,
        rank=i+4
    )
    for i in range(N_ACTORS)
]
