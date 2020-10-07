from genrl.distributed import (
    Master,
    ExperienceServer,
    ParameterServer,
    ActorNode,
    LearnerNode,
    DistributedTrainer,
    WeightHolder,
)
from genrl.core import ReplayBuffer
from genrl.agents import DDPG
import gym
import argparse
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int)
args = parser.parse_args()

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
        next_obs, reward, done, info = agent.env.step(action)
        print("Sending experience")
        experience_server_rref.rpc_sync().push((obs, action, reward, done, next_obs))
        print("Done sending experience")
        obs = next_obs
        if done:
            break


class MyTrainer(DistributedTrainer):
    def __init__(self, agent, train_steps, batch_size):
        super(MyTrainer, self).__init__(agent)
        self.train_steps = train_steps
        self.batch_size = batch_size

    def train(self, parameter_server_rref, experience_server_rref):
        print("IN TRAIN")
        for i in range(self.train_steps):
            print("GETTING BATCH")
            batch = experience_server_rref.rpc_sync().sample(self.batch_size)
            print("GOT BATCH")
            if batch is None:
                continue
            self.agent.update_params(batch)
            print("Storing weights")
            parameter_server_rref.rpc_sync().store_weights(self.agent.get_weights())
            # remote_method(WeightHolder.store_weights, parameter_server_rref, self.agent.get_weights())
            print("Done storing weights")
            print(f"TRAINER: {i} STESPS done")


mp.set_start_method("fork")

master = Master(world_size=6, address="localhost", port=29504)
env = gym.make("Pendulum-v0")
agent = DDPG("mlp", env)
parameter_server = ParameterServer(
    "param-0", master, WeightHolder(agent.get_weights()), rank=1
)
buffer = ReplayBuffer(BUFFER_SIZE)
experience_server = ExperienceServer("experience-0", master, buffer, rank=2)
trainer = MyTrainer(agent, TRAIN_STEPS, BATCH_SIZE)
learner = LearnerNode(
    "learner-0", master, parameter_server, experience_server, trainer, rank=3
)
actors = [
    ActorNode(
        f"actor-{i}",
        master,
        parameter_server,
        experience_server,
        learner,
        agent,
        collect_experience,
        rank=i + 4,
    )
    for i in range(N_ACTORS)
]
