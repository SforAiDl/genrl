import torch
import torch.nn as nn
import torch.nn.functional as F

# Centralized Policy Value Network
# class CentralizedActorCritic(nn.Module): 

#     def __init__(self, obs_dim, action_dim):
#         super(CentralizedActorCritic, self).__init__()

#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#         self.shared_layer = nn.Linear(self.obs_dim, 256)
#         # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.shared_layer.weight)

#         self.value = nn.Linear(256, 1)
#         # torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.value.weight)
        
#         self.policy = nn.Linear(256, self.action_dim)
#         # torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')
#         torch.nn.init.xavier_uniform_(self.policy.weight)

#     def forward(self, x):
#         x_s = F.relu(self.shared_layer(x))
#         qval = self.value(x_s)
#         policy = self.policy(x_s)

#         return policy,qval

# Trying more neurons with an extra layer 
class CentralizedActorCritic(nn.Module): 

    def __init__(self, obs_dim, action_dim):
        super(CentralizedActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.shared_layer_1 = nn.Linear(self.obs_dim, 512)
        # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.shared_layer_1.weight)
        # torch.nn.init.xavier_normal_(self.shared_layer_1.weight)

        self.shared_layer_2 = nn.Linear(512, 256)
        # torch.nn.init.kaiming_normal_(self.shared_layer.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.shared_layer_2.weight)
        # torch.nn.init.xavier_normal_(self.shared_layer_2.weight)

        self.value = nn.Linear(256, 1)
        # torch.nn.init.kaiming_normal_(self.value.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.value.weight)
        # torch.nn.init.xavier_normal_(self.value.weight)

        self.policy = nn.Linear(256, self.action_dim)
        # torch.nn.init.kaiming_normal_(self.policy.weight, mode='fan_in')
        torch.nn.init.xavier_uniform_(self.policy.weight)
        # torch.nn.init.xavier_normal_(self.policy.weight)

    def forward(self, x):
        x_s = F.relu(self.shared_layer_1(x))
        x_s = F.relu(self.shared_layer_2(x_s))
        qval = self.value(x_s)
        policy = self.policy(x_s)

        return policy,qval

