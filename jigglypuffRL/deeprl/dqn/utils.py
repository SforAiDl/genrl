import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

class DQN_Mlp(nn.Module):
    def __init__(self, env):
        super(DQN_Mlp, self).__init__()
        
        self.env = env
        
        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if np.random.rand() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(self.env.action_space.n)
        return action
    
# TO DO add CNN Networks 