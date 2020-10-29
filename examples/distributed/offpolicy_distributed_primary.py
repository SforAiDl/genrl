from genrl.distributed import (
    Master,
    ExperienceServer,
    ParameterServer,
    ActorNode,
    LearnerNode,
)
from genrl.core import ReplayBuffer
from genrl.agents import DDPG
from genrl.trainers import DistributedTrainer
from genrl.utils import Logger
import gym
import torch.distributed.rpc as rpc
import time


N_ACTORS = 1
BUFFER_SIZE = 5000
MAX_ENV_STEPS = 500
TRAIN_STEPS = 5000
BATCH_SIZE = 64
INIT_BUFFER_SIZE = 1000
WARMUP_STEPS = 1000


def collect_experience(agent, parameter_server, experience_server, learner):
    while not learner.is_completed():
        agent.load_weights(parameter_server.get_weights())
        obs = agent.env.reset()
        done = False
        for i in range(MAX_ENV_STEPS):
            action = (
                agent.env.action_space.sample()
                if i < WARMUP_STEPS
                else agent.select_action(obs)
            )
            next_obs, reward, done, _ = agent.env.step(action)
            experience_server.push((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                break


class MyTrainer(DistributedTrainer):
    def __init__(self, agent, train_steps, batch_size, init_buffer_size, log_interval=200):
        super(MyTrainer, self).__init__(agent)
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.init_buffer_size = init_buffer_size
        self.logger = Logger(formats=["stdout"])
        self.log_interval = log_interval

    def train(self, parameter_server, experience_server):
        while experience_server.__len__() < self.init_buffer_size:
            time.sleep(1)
        for i in range(self.train_steps):
            batch = experience_server.sample(self.batch_size)
            if batch is None:
                continue
            self.agent.update_params(1, batch)
            parameter_server.store_weights(self.agent.get_weights())
            if i % self.log_interval == 0:
                self.evaluate(i)


master = Master(
    world_size=5,
    address="localhost",
    port=29502,
    proc_start_method="fork",
    rpc_backend=rpc.BackendType.TENSORPIPE,
)
env = gym.make("Pendulum-v0")
agent = DDPG("mlp", env)
parameter_server = ParameterServer("param-0", master, agent.get_weights(), rank=1)
buffer = ReplayBuffer(BUFFER_SIZE)
experience_server = ExperienceServer("experience-0", master, buffer, rank=2)
trainer = MyTrainer(agent, TRAIN_STEPS, BATCH_SIZE, INIT_BUFFER_SIZE)
learner = LearnerNode("learner-0", master, "param-0", "experience-0", trainer, rank=3)
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
