import random
from collections import namedtuple
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from genrl.deep.common.utils import get_env_properties
from genrl.deep.common.buffers import HindsightMemory

class FNN(nn.Module):
    
    def __init__(self,state_size,num_actions):
        super(FNN, self).__init__()
        self.ln1 = nn.Linear(state_size,256)
        self.ln2 = nn.Linear(256,num_actions)
        
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x



class DuelingMLP(nn.Module):

    def __init__(self, state_size, num_actions):
        super().__init__()
        self.linear = nn.Linear(state_size, 256)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.unsqueeze(0) if len(x.size()) == 1 else x
        x = F.relu(self.linear(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        action_values = (value + (advantage - advantage.mean(dim=1, keepdim=True))).squeeze()
        return action_values

class DQNwithHER():

    def __init__(self,env,device,dueling=False,gamma=0.98,batch_size=128,train_threshold=1000,mem_cap=1e6):

        state_size,num_actions,_,_=get_env_properties(env)
        state_size=2*state_size
        self.Experience = namedtuple("Experience", field_names="state action reward next_state done")
        self.state_size = state_size
        self.num_actions = num_actions
        self.env=env
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_start = train_threshold
        self.device=device

        self.memory = HindsightMemory(int(mem_cap))
        if dueling==False:
            self.Q_network=FNN(state_size,num_actions)
            self.Q_network=self.Q_network.to(self.device)
            self.target_network=FNN(state_size,num_actions)
            self.target_network=self.target_network.to(self.device)
        else:
            self.Q_network=DuelingMLP(state_size,num_actions)
            self.Q_network=self.Q_network.to(self.device)
            self.target_network=DuelingMLP(state_size,num_actions)
            self.target_network=self.target_network.to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=0.001)

    def push_experience(self, state, action, reward, next_state, done):
        self.memory.push(self.Experience(state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def take_action(self, state, epsilon):
        if random.random() > epsilon:
            state=state.to(self.device)
            return self.greedy_action(state)
        else:
            return torch.randint(self.num_actions, size=()).to(self.device)

    def greedy_action(self, state):
        with torch.no_grad():
            state=state.to(self.device)
            return self.Q_network(state).argmax()

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = self.Experience(*zip(*experiences))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        reward_batch=reward_batch.to(self.device)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])

        state_batch=state_batch.to(self.device)
        non_final_next_states=non_final_next_states.to(self.device)
        Q_values = self.Q_network(state_batch)[range(self.batch_size), action_batch]

        # Double DQN target #
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        number_of_non_final = sum(non_final_mask)
        with torch.no_grad():
            argmax_actions = self.Q_network(non_final_next_states).argmax(1)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
                range(number_of_non_final), argmax_actions]

        Q_targets = reward_batch + self.gamma * next_state_values

        assert Q_values.shape == Q_targets.shape

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        self.optimizer.step()

