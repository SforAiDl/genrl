import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.autograd as autograd
from typing import Tuple, Type, Union
import numpy as np



class SharedMLP(nn.Module):
    def __init__(self, network1_prev,network2_prev,shared,network1_post,network2_post,weight_init,activation_func):
        super(SharedMLP, self).__init__()

        if len(network1_prev) != 0:
            self.network1_prev = nn.ModuleList()
        if len(network2_prev) != 0:
            self.network2_prev = nn.ModuleList()
        if len(shared) != 0:
            self.shared = nn.ModuleList()
        if len(network1_post) != 0:
            self.network1_post = nn.ModuleList()
        if len(network2_post) != 0:
            self.network2_post = nn.ModuleList()

        # add more activation functions
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = None

        # add more weight init
        if weight_init == "xavier_uniform":
            self.weight_init = torch.nn.init.xavier_uniform_
        elif weight_init == "xavier_normal":
            self.weight_init = torch.nn.init.xavier_normal_
        else:
            self.weight_init = None

        if len(shared) != 0 or len(network1_post) != 0 or len(network2_post) != 0:
            if not (network1_prev[-1]==network2_prev[-1] and network1_prev[-1]==shared[0] and network1_post[0]==network2_post[0] and network1_post[0]==shared[-1]):
                raise ValueError


        for i in range(len(network1_prev)-1):
            self.network1_prev.append(nn.Linear(network1_prev[i],network1_prev[i+1]))
            if self.weight_init is not None:
                self.weight_init(self.network1_prev[-1].weight)

        for i in range(len(network2_prev)-1):
            self.network2_prev.append(nn.Linear(network2_prev[i],network2_prev[i+1]))
            if self.weight_init is not None:
                self.weight_init(self.network2_prev[-1].weight)

        for i in range(len(shared)-1):
            self.shared.append(nn.Linear(shared[i], shared[i+1]))
            if self.weight_init is not None:
                self.weight_init(self.shared[-1].weight)

        for i in range(len(network1_post)-1):
            self.network1_post.append(nn.Linear(network1_post[i],network1_post[i+1]))
            if self.weight_init is not None:
                self.weight_init(self.network1_post[-1].weight)

        for i in range(len(network2_post)-1):
            self.network2_post.append(nn.Linear(network2_post[i],network2_post[i+1]))
            if self.weight_init is not None:
                self.weight_init(self.network2_post[-1].weight)




class MLP(nn.Module):
    def __init__(self,layer_sizes,weight_init,activation_func,concat_ind):
        super(MLP, self).__init__()

        self.model = nn.ModuleList()

        # add more activation functions
        if activation_function == "relu":
            self.activation = F.relu
        elif activation_function == "tanh":
            self.activation = torch.tanh

        # add more weight init
        if weight_init == "xavier_uniform":
            self.weight_init = torch.nn.init.xavier_uniform_
        elif weight_init == "xavier_normal":
            self.weight_init = torch.nn.init.xavier_normal_


        for i in range(len(layer_sizes)-1):
            if i==concat_ind:
                i=i+1
            self.model.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            self.weight_init(self.model[-1])





