from typing import Tuple

import torch
import torch.nn as nn


from genrl.utils import get_obs_network, get_distribution, normalize_tensor, compute_advantage
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer
from genrl.losses import VanillaPolicyGradientLoss, TDErrorLoss

env = VectorEnv("CartPole-v1", 2)

obs_network = get_obs_network(env, 64)
# obs_network = nn.Linear(4, 64)

# gnn = nn.Linear()

distribution = get_distribution(env)
# Policy

mlp = nn.Linear(64, 64)

vpg_agent = OnPolicyAgent(nn.Sequential([obs_network, mlp, distribution, VanillaPolicyGradientLoss]), )
# 1. Shared Actor Critics?


# Actor Critic 

trainer = OnPolicyTrainer(vpg_agent, env)
trainer.train()

"""
Idea is to add composability of core code with rest of Pytorch.
"""

"""
For A2C, since we make use of advantages, we need to add a custom loss.
"""

class Agent():
    def __init__(self):
        pass
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        pass
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor):
        pass


class A2CPolicyGradientLoss(nn.Module):
    def __init__(self):
        super(A2CPolicyGradientLoss).__init__()

    def forward(self, agent, rollout):
        advantages = compute_advantage(rollout)
        advantages = normalize_tensor(advantages)

        values, log_probs, entropy = agent.evaluate_action(rollout.obs, rollout.action)

        policy_loss = advantages * rollout.log_probs
        return -1*torch.mean(policy_loss)


vpg_agent = OnPolicyActorCriticAgent(nn.Sequential([obs_network, mlp, distribution, A2CPolicyGradientLoss]), nn.Sequential([obs_network, mlp, distribution, ValueNetwork, TDErrorLoss]))
trainer = OnPolicyTrainer(vpg_agent, env)
trainer.train()


class PPOPolicyGradientLoss(nn.Module):
    def __init__(self):
        super(VanillaPolicyGradientLoss).__init__()

    def forward(self, agent, rollout):
        advantages = compute_advantage(rollout)
        advantages = normalize_tensor(advantages)

        values, log_probs, entropy = agent.evaluate_action(rollout.obs, rollout.action)

        ratio = torch.exp(log_probs - rollout.log_probs)

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1 - 0.2, 1 + 0.2
        )
        policy_loss = torch.min(policy_loss_1, policy_loss_2).mean()
        
        return -1*policy_loss


ppo_agent = OnPolicyActorCriticAgent(nn.Sequential([obs_network, mlp, distribution, PPOPolicyGradientLoss]), nn.Sequential([obs_network, mlp, distribution, ValueNetwork, TDErrorLoss]))
trainer = OnPolicyTrainer(vpg_agent, env)
trainer.train()


"""Values
Values are observation Networks.
"""

"""Policies
Policies just need a head depending on the distribution -- can be figured out using the environment.
"""

"""Distributional
Have a head for distributional forward
"""

"""Distributed
Learners.
Actors (Observation Networks) --> MultiAgent
"""

"""Offline
Learners.
"""


"""OnPolicyTrainer
Rollouts collection
Make changes over Trajectory Data.
Compute Loss and Backprop
"""

"""OffPolicyTrainer
Observe
Update
"""



"""
Off Policy Agents

Select Action.

Sample 


"""

class 



ddpg = ActorCriticAgent(nn.Sequential([obs_network, mlp, distribution, VanillaPolicyGradientLoss]))

