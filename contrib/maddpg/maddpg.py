import torch
import numpy as np

from agent import DDPGAgent
from utils import MultiAgentReplayBuffer


class MADDPG:

    def __init__(self, env, buffer_maxlen):
        self.env = env
        self.num_agents = env.n
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.agents = [DDPGAgent(self.env, i) for i in range(self.num_agents)]

    def get_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i])
            actions.append(action)
        return actions

    def update(self, batch_size):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
            global_state_batch, global_actions_batch, global_next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        
        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]

            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.FloatTensor(next_obs_batch_i)
                indiv_next_action = agent.actor.forward(next_obs_batch_i)
                indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for indiv_next_action_j in indiv_next_action]
                indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)
            next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)

            self.agents[i].update(indiv_reward_batch_i, obs_batch_i, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions)
            self.agents[i].target_update()
    
    def run(self, max_episode, max_steps, batch_size):
        episode_rewards = []
        for episode in range(max_episode):
            states = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                actions = self.get_actions(states)
                next_states, rewards, dones, _ = self.env.step(actions)
                episode_reward += np.mean(rewards)
        
                if all(dones) or step == max_steps - 1:
                    dones = [1 for _ in range(self.num_agents)]
                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    episode_rewards.append(episode_reward)
                    print("episode: {}  |  reward: {}  \n".format(episode, np.round(episode_reward, decimals=4)))
                    break
                else:
                    dones = [0 for _ in range(self.num_agents)]
                    self.replay_buffer.push(states, actions, rewards, next_states, dones)
                    states = next_states 

                    if len(self.replay_buffer) > batch_size:
                        self.update(batch_size)