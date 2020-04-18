import torch
import torch.nn as nn
import torch.nn.functional as F

class TDLoss(nn.SmoothL1Loss):
    def __init__(self, actor, critic, target_critic, huber_loss=False,
                 stochastic=False, prioritized_replay_buffer_weight=None, 
                 alpha=None, *args, **kwargs):
        super(TDLoss, self).__init__(args, kwargs)
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.huber_loss = huber_loss
        self.stochastic = stochastic
        self.prioritized_replay_buffer_weight = prioritized_replay_buffer_weight # Only used in DQN
        self.alpha = alpha

        if self.stochastic:
            if self.alpha is None:
                raise Exception("Alpha cannot be None is loss is for Stochastic Policies")

    def forward(self, next_state, state, action, reward, done):
        if self.stochastic:
            next_action, next_log_pi, _ = self.actor.get_action(state)
        else:
            next_action = self.actor.get_action(state)
        
        inp = torch.cat([next_state, next_action], dim=-1)
        target_q = self.target_critic.get_value(inp)
        
        if self.stochastic:
            target_q -= self.alpha * next_log_pi
        
        next_q = reward + self.gamma * (1 - done) * target_q

        Q = self.critic.get_value(torch.cat([state, action], dim=-1))

        if self.huber_loss:
            loss = F.smooth_l1_loss(Q, next_q)
        else:
            loss = F.mse_loss(Q, next_q)
        
        return loss


class TDLosswithMultipleCritics(nn.SmoothL1Loss):
    def __init__(self, actor, critics, target_critics, huber_loss=False, 
                 stochastic=False,  prioritized_replay_buffer_weight=None, 
                 alpha=None, *args, **kwargs):
        super(TDLoss, self).__init__(actor, critics, target_critics, stochastic, 
                                     prioritized_replay_buffer_weight, alpha, args, kwargs)
        self.actor = actor
        self.critics = critics
        self.target_critics = target_critics
        self.huber_loss = huber_loss
        self.stochastic = stochastic
        self.prioritized_replay_buffer_weight = prioritized_replay_buffer_weight # Only used in DQN
        self.alpha = alpha

        if self.stochastic:
            if self.alpha is None:
                raise Exception("Alpha cannot be None is loss is for Stochastic Policies")
        
        if isinstance(self.critics, nn.Module):
            raise Exception("Critics should be a list of nn.Modules. Use TDLoss instead")
        if isinstance(self.target_critics, nn.Module):
            raise Exception("Target Critics should be a list of nn.Modules. Use TDLoss instead")

    def forward(self, decideQ, next_state, state, action, reward, done):
        if self.stochastic:
            next_action, next_log_pi, _ = self.actor.get_action(state)
        else:
            next_action = self.actor.get_action(state)
        
        inp = torch.cat([next_state, next_action], dim=-1)
        target_Qs = []
        for target_critic in self.target_critics:
            target_q = self.target_critic.get_value(inp)
            target_Qs.append(target_q)
        
        target_q = decideQ(target_Qs)
        if self.stochastic:
            target_q -= self.alpha * next_log_pi

        next_q = reward + self.gamma * (1 - done) * target_q

        losses = []
        for critic in self.critics:
            Q = self.critic.get_value(torch.cat([state, action], dim=-1))
            if self.huber_loss:
                loss = F.smooth_l1_loss(Q, next_q)
            else:
                loss = F.mse_loss(Q, next_q)
            losses.append(loss)
        return *losses
        