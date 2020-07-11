import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from a2c import *
import torch.nn.functional as F
import gc

class A2CAgent:

  def __init__(self,env,lr=2e-4,gamma=0.99):
    self.env = env
    self.lr = lr
    self.gamma = gamma

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.num_agents = self.env.n

    self.input_dim = env.observation_space[0].shape[0]
    self.action_dim = self.env.action_space[0].n
    self.actorcritic = CentralizedActorCritic(self.input_dim,self.action_dim).to(self.device)
    model_path = ".tests/models/four_agents/actorcritic_network_no_comms_discounted_rewards_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt"
    self.actorcritic.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    self.MSELoss = nn.MSELoss()
    self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(),lr=lr)

  def get_action(self,state):
    state = torch.FloatTensor(state).to(self.device)
    logits,_ = self.actorcritic.forward(state)
    # print("logits:",logits)
    del state
    dist = F.softmax(logits,dim=0)
    # print("dist:",dist)
    del logits
    # print('dist: ', dist)
    probs = Categorical(dist)
    # print("PROBS:",probs)
    index = probs.sample().cpu().detach().item()
    
    return index


  def update(self,global_state_batch,global_next_state_batch,global_actions_batch,rewards):
    

    curr_logits,curr_Q = self.actorcritic.forward(global_state_batch)
    # _,next_Q = self.actorcritic.forward(global_next_state_batch)
    # estimated_Q = rewards.unsqueeze(dim=2) + self.gamma*next_Q
    # print("REWARDS",rewards.shape)
    discounted_rewards = np.asarray([[torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[k][j:].size(0))])* rewards[k][j:]) for j in range(rewards.size(0))] for k in range(self.num_agents)])
    discounted_rewards = np.transpose(discounted_rewards)
    # print("DISCOUNTED REWARDS",discounted_rewards.shape)
    value_targets = rewards + torch.FloatTensor(discounted_rewards).to(self.device)
    # print("VALUE TARGETS:",value_targets.shape)
    value_targets = value_targets.unsqueeze(dim=-1)
    # print("VALUE TARGETS:",value_targets.shape)


    # print("CURRENT Q")
    # print(curr_Q)
    # print(curr_Q.shape)
    # print("ESTIMATES Q")
    # print(estimated_Q)
    # print(estimated_Q.shape)
    # print("CURRENT LOGITS")
    # print(curr_logits)
    # print(curr_logits.shape)

    

    # critic_loss = self.MSELoss(curr_Q,estimated_Q.detach())
    # critic_loss = F.smooth_l1_loss(curr_Q,estimated_Q.detach())
    critic_loss = F.smooth_l1_loss(curr_Q,value_targets)
    # print("CRITIC LOSS")
    # print(critic_loss)

    dists = F.softmax(curr_logits,dim=-1)
    probs = Categorical(dists)

    # entropy = []
    # for dist in dists:
    #   entropy.append(-torch.sum(dist*torch.log(torch.clamp(dist,1e-10,1))))
    # entropy = torch.stack(entropy).mean()
    entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=2))
    # print("ENTROPY")
    # print(entropy)

    # advantage = estimated_Q - curr_Q
    advantage = value_targets - curr_Q
    # print("ACTIONS")
    # print(global_actions_batch)
    # print(global_actions_batch.shape)
    # print(probs.log_prob(global_actions_batch).shape)
    policy_loss = -probs.log_prob(global_actions_batch).unsqueeze(dim=-1) * advantage.detach()
    policy_loss = policy_loss.mean()

    total_loss = policy_loss + critic_loss - 0.008*entropy

    self.actorcritic_optimizer.zero_grad()
    total_loss.backward(retain_graph=False)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(),0.5)
    self.actorcritic_optimizer.step()
    

    return critic_loss,policy_loss,entropy,total_loss,grad_norm
