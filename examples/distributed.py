from genrl.agents import DDPG
from genrl.trainers import OffPolicyTrainer
from genrl.trainers.distributed import DistributedOffPolicyTrainer
from genrl.environments import VectorEnv
import gym
import reverb
import numpy as np
import multiprocessing as mp
import threading


# env = VectorEnv("Pendulum-v0")
# agent = DDPG("mlp", env)
# trainer = OffPolicyTrainer(agent, env)
# trainer.train()


env = gym.make("Pendulum-v0")
agent = DDPG("mlp", env)

# o = env.reset()
# action = agent.select_action(o)
# next_state, reward, done, info = env.step(action.numpy())

# buffer_server = reverb.Server(
#     tables=[
#         reverb.Table(
#             name="replay_buffer",
#             sampler=reverb.selectors.Uniform(),
#             remover=reverb.selectors.Fifo(),
#             max_size=10,
#             rate_limiter=reverb.rate_limiters.MinSize(4),
#         )
#     ],
#     port=None,
# )
# client = reverb.Client(f"localhost:{buffer_server.port}")
# print(client.server_info())

# state = env.reset()
# action = agent.select_action(state)
# next_state, reward, done, info = env.step(action.numpy())

# state = next_state.copy()
# print(client.server_info())
# print("going to insert")
# client.insert([state, action, np.array([reward]), np.array([done]), next_state], {"replay_buffer": 1})
# client.insert([state, action, np.array([reward]), np.array([done]), next_state], {"replay_buffer": 1})
# client.insert([state, action, np.array([reward]), np.array([done]), next_state], {"replay_buffer": 1})
# client.insert([state, action, np.array([reward]), np.array([done]), next_state], {"replay_buffer": 1})
# client.insert([state, action, np.array([reward]), np.array([done]), next_state], {"replay_buffer": 1})
# print("inserted")

# # print(list(client.sample('replay_buffer', num_samples=1)))


# def sample(address):
#     print("-- entered proc")
#     client = reverb.Client(address)
#     print("-- started client")
#     print(list(client.sample('replay_buffer', num_samples=1)))

# a = f"localhost:{buffer_server.port}"
# print("create process")
# # p = mp.Process(target=sample, args=(a,))
# p = threading.Thread(target=sample, args=(a,))
# print("start process")
# p.start()
# print("wait process")
# p.join()
# print("end process")

trainer = DistributedOffPolicyTrainer(agent, env)
trainer.train(
    n_actors=2, max_buffer_size=100, batch_size=4, max_updates=10, update_interval=1
)
