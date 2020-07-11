#import matplotlib
#import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from a2c_agent import A2CAgent
import gc


class MAA2C:

  def __init__(self,env):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.env = env
    self.num_agents = env.n
    self.agents = A2CAgent(self.env)


    self.writer = SummaryWriter('tests/longer_runs/four_agents/simple_spread_2_shared_layers_no_comms_discounted_rewards_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs_continued')

  def get_actions(self,states):
    actions = []
    for i in range(self.num_agents):
      action = self.agents.get_action(states[i])
      actions.append(action)
    return actions

  def update(self,trajectory,episode):

    states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
    next_states = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
    actions = torch.LongTensor([sars[2] for sars in trajectory]).to(self.device)
    rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)

    critic_loss,policy_loss,entropy,total_loss,grad_norm = self.agents.update(states,next_states,actions,rewards)

    self.writer.add_scalar('Loss/Entropy loss',entropy,episode)
    self.writer.add_scalar('Loss/Value Loss',critic_loss,episode)
    self.writer.add_scalar('Loss/Policy Loss',policy_loss,episode)
    self.writer.add_scalar('Loss/Total Loss',total_loss,episode)
    self.writer.add_scalar('Gradient Normalization/Grad Norm',grad_norm,episode)




  def run(self,max_episode,max_steps):
    for episode in range(max_episode):
      states = self.env.reset()
      trajectory = []
      episode_reward = 0
      for step in range(max_steps):
        actions = self.get_actions(states)
        next_states,rewards,dones,info = self.env.step(actions)

        # print(rewards)


        episode_reward += np.sum(rewards)

        # print(dones)
        # print(all(dones))

        if all(dones) or step == max_steps-1:
          # print("STEP",step)
          # print(all(dones))

          dones = [1 for _ in range(self.num_agents)]
          trajectory.append([states,next_states,actions,rewards])
          print("*"*100)
          print("EPISODE: {} | REWARD: {} \n".format(episode,np.round(episode_reward,decimals=4)))
          print("*"*100)
          self.writer.add_scalar('Reward Incurred/Length of the episode',step,episode)
          self.writer.add_scalar('Reward Incurred/Reward',episode_reward,episode)
          break
        else:
          dones = [0 for _ in range(self.num_agents)]
          trajectory.append([states,next_states,actions,rewards])
          states = next_states
      
#       make a directory called models
      if episode%500:
        torch.save(self.agents.actorcritic.state_dict(), ".tests//models/four_agents/actorcritic_network_no_comms_discounted_rewards_lr_2e-4_with_grad_norm_0.5_entropy_pen_0.008_xavier_uniform_init_clamp_logs.pt")
      
        
      self.update(trajectory,episode) 

      # torch.cuda.empty_cache()

      # for obj in gc.get_objects():
      #   try:
      #       if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      #           print(type(obj), obj.size())
      #   except: pass

