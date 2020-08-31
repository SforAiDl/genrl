import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np 

from model import CentralizedCritic, Actor


class DDPGAgent:

    def __init__(self, env, agent_id, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2):
        self.env = env
        self.agent_id = agent_id
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau

        self.device = "cpu"
        self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     self.device = "cuda"

        self.obs_dim = self.env.observation_space[agent_id].shape[0]
        self.action_dim = self.env.action_space[agent_id].n
        self.num_agents = self.env.n

        self.critic_input_dim = int(np.sum([env.observation_space[agent].shape[0] for agent in range(env.n)]))
        self.actor_input_dim = self.obs_dim

        self.critic = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.actor = Actor(self.actor_input_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.actor_input_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().squeeze(0)).to(self.device)
        action = self.actor.forward(state)
        action = self.onehot_from_logits(action)

        return action
    
    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
            range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
        # chooses between best and random actions using epsilon greedy
        return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                            enumerate(torch.rand(logits.shape[0]))])
    
    def update(self, indiv_reward_batch, indiv_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions):
        """
        indiv_reward_batch      : only rewards of agent i
        indiv_obs_batch         : only observations of agent i
        global_state_batch      : observations of all agents are concatenated
        global actions_batch    : actions of all agents are concatenated
        global_next_state_batch : observations of all agents are concatenated
        next_global_actions     : actions of all agents are concatenated
        """
        indiv_reward_batch = torch.FloatTensor(indiv_reward_batch).to(self.device)
        indiv_reward_batch = indiv_reward_batch.view(indiv_reward_batch.size(0), 1).to(self.device) 
        indiv_obs_batch = torch.FloatTensor(indiv_obs_batch).to(self.device)          
        global_state_batch = torch.FloatTensor(global_state_batch).to(self.device)    
        global_actions_batch = torch.stack(global_actions_batch).to(self.device)      
        global_next_state_batch = torch.FloatTensor(global_next_state_batch).to(self.device)
        next_global_actions = next_global_actions

        # update critic        
        self.critic_optimizer.zero_grad()
        
        curr_Q = self.critic.forward(global_state_batch, global_actions_batch)
        next_Q = self.critic_target.forward(global_next_state_batch, next_global_actions)
        estimated_Q = indiv_reward_batch + self.gamma * next_Q
        
        critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()

        policy_loss = -self.critic.forward(global_state_batch, global_actions_batch).mean()
        curr_pol_out = self.actor.forward(indiv_obs_batch)
        policy_loss += -(curr_pol_out**2).mean() * 1e-3 
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
    
    def target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))        