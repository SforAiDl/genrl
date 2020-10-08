from genrl.distributed import (
    Master,
    ExperienceServer,
    ParameterServer,
    ActorNode,
    LearnerNode,
    WeightHolder,
)
from genrl.core import ReplayBuffer
from genrl.agents import DDPG
from genrl.trainers import DistributedTrainer
import gym
import torch.multiprocessing as mp


N_ACTORS = 2
BUFFER_SIZE = 10
MAX_ENV_STEPS = 100
TRAIN_STEPS = 50
BATCH_SIZE = 1


def collect_experience(agent, experience_server_rref):
    obs = agent.env.reset()
    done = False
    for i in range(MAX_ENV_STEPS):
        action = agent.select_action(obs)
        next_obs, reward, done, info = agent.env.step(action)
        experience_server_rref.rpc_sync().push((obs, action, reward, next_obs, done))
        obs = next_obs
        if done:
            break


class MyTrainer(DistributedTrainer):
    def __init__(self, agent, train_steps, batch_size):
        super(MyTrainer, self).__init__(agent)
        self.train_steps = train_steps
        self.batch_size = batch_size

    def train(self, parameter_server_rref, experience_server_rref):
        i = 0
        while i < self.train_steps:
            batch = experience_server_rref.rpc_sync().sample(self.batch_size)
            if batch is None:
                continue
            self.agent.update_params(batch, 1)
            parameter_server_rref.rpc_sync().store_weights(self.agent.get_weights())
            print(f"Trainer: {i + 1} / {self.train_steps} steps completed")
            i += 1


mp.set_start_method("fork")

master = Master(world_size=8, address="localhost", port=29500)
env = gym.make("Pendulum-v0")
agent = DDPG("mlp", env)
parameter_server = ParameterServer(
    "param-0", master, WeightHolder(agent.get_weights()), rank=1
)
buffer = ReplayBuffer(BUFFER_SIZE)
experience_server = ExperienceServer("experience-0", master, buffer, rank=2)
trainer = MyTrainer(agent, TRAIN_STEPS, BATCH_SIZE)
learner = LearnerNode(
    "learner-0", master, "param-0", "experience-0", trainer, rank=3
)
actors = [
    ActorNode(
        name=f"actor-{i}",
        master=master,
        parameter_server_name="param-0",
        experience_server_name="experience-0",
        learner_name="learner-0",
        agent=agent,
        collect_experience=collect_experience,
        rank=i + 4,
    )
    for i in range(N_ACTORS)
]
