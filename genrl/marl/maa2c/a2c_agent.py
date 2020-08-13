import gc

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from a2c import *
from torch.autograd import Variable
from torch.distributions import Categorical


class A2CAgent:
    def __init__(self, env, lr=2e-4, gamma=0.99, load_model=None, entropy_weight=0.008):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.w = entropy_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_agents = self.env.n

        self.input_dim = self.env.observation_space[0].shape[0]
        self.action_dim = self.env.action_space[0].n

        self.actorcritic = None
        self.optimizer = None

        self.episode_reward = 0
        self.steps_per_episode = None
        self.final_step = 0

        self.values = None
        self.logits = None
        self.states = None
        self.next_states = None
        self.actions = None
        self.dones = None
        self.rewards = None

        self.value_loss = None
        self.policy_loss = None
        self.entropy = None
        self.total_loss = None

        self.grad_norm = None

        self.setup_model(load_model)

    def setup_model(self, load_model):

        self.actorcritic = CentralizedActorCritic(self.input_dim, self.action_dim).to(
            self.device
        )
        if load_model is not None:
            self.actorcritic.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

        self.optimizer = optim.Adam(self.actorcritic.parameters(), lr=self.lr)

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.actorcritic.get_action(states[i], self.device, False)
            actions.append(action)
        return actions

    def collect_rollouts(self, states):
        trajectory = []
        self.episode_reward = 0
        self.final_step = 0
        current_states = states

        for step in range(self.steps_per_episode):

            actions = self.get_actions(current_states)
            next_states, rewards, dones, info = self.env.step(actions)
            self.episode_reward += np.sum(rewards)

            if all(dones) or step == self.steps_per_episode - 1:

                dones = [1 for _ in range(self.num_agents)]
                trajectory.append([states, next_states, actions, rewards, dones])
                print("REWARD: {} \n".format(np.round(self.episode_reward, decimals=4)))
                print("*" * 100)
                self.final_step = step
                break
            else:
                dones = [0 for _ in range(self.num_agents)]
                trajectory.append([states, next_states, actions, rewards, dones])
                current_states = next_states
                self.final_step = step

        self.states = torch.FloatTensor([sars[0] for sars in trajectory]).to(
            self.device
        )
        self.next_states = torch.FloatTensor([sars[1] for sars in trajectory]).to(
            self.device
        )
        self.actions = torch.LongTensor([sars[2] for sars in trajectory]).to(
            self.device
        )
        self.rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(
            self.device
        )
        self.dones = torch.LongTensor([sars[4] for sars in trajectory])

        self.logits, self.values = self.actorcritic.forward(self.states)

        return self.values, self.dones

    def get_traj_loss(self, curr_Q, done):
        discounted_rewards = np.asarray(
            [
                [
                    torch.sum(
                        torch.FloatTensor(
                            [
                                self.gamma ** i
                                for i in range(self.rewards[k][j:].size(0))
                            ]
                        )
                        * self.rewards[k][j:]
                    )
                    for j in range(self.rewards.size(0))
                ]
                for k in range(self.num_agents)
            ]
        )
        discounted_rewards = np.transpose(discounted_rewards)
        value_targets = self.rewards + torch.FloatTensor(discounted_rewards).to(
            self.device
        )
        value_targets = value_targets.unsqueeze(dim=-1)
        self.value_loss = F.smooth_l1_loss(curr_Q, value_targets)

        dists = F.softmax(self.logits, dim=-1)
        probs = Categorical(dists)

        self.entropy = -torch.mean(
            torch.sum(dists * torch.log(torch.clamp(dists, 1e-10, 1.0)), dim=-1)
        )

        advantage = value_targets - curr_Q
        self.policy_loss = -probs.log_prob(self.actions) * advantage.detach()
        self.policy_loss = self.policy_loss.mean()

        self.total_loss = self.policy_loss + self.value_loss - self.w * self.entropy

    def update_policy(self):
        self.optimizer.zero_grad()
        self.total_loss.backward(retain_graph=False)
        self.grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actorcritic.parameters(), 0.5
        )
        self.optimizer.step()

    def get_logging_params(self):
        logging_params = {
            "Loss/Entropy loss": self.entropy.item(),
            "Loss/Value Loss": self.value_loss.item(),
            "Loss/Policy Loss": self.policy_loss,
            "Loss/Total Loss": self.total_loss,
            "Gradient Normalization/Grad Norm": self.grad_norm,
            "Reward Incurred/Length of the episode": self.final_step,
            "Reward Incurred/Reward": self.episode_reward,
        }
        return logging_params

    def get_hyperparams(self):
        hyperparams = {
            "gamma": self.gamma,
            "entropy_weight": self.w,
            "lr_actor": self.lr,
            "lr_critic": self.lr,
            "actorcritic_weights": self.actorcritic.state_dict(),
        }

        return hyperparams
