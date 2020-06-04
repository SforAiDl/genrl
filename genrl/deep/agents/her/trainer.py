import random
from collections import namedtuple
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from genrl.deep.common.utils import get_env_properties
from genrl.deep.common.buffer import HindsightMemory

class HERTrainer():
  def __init__(self,agent):
    self.agent=agent
    self.env=self.agent.env
    
  
  def train(self,num_epochs=30,num_cycles=50,num_episodes=16,future_k=4,num_opt_steps=40,HER=True,eps_max=0.2,eps_min=0.0,exploration_fraction=0.5):

    #env = BitFlipEnvironment(num_bits)

    #num_actions = num_bits
    #state_size = 2 * num_bits
    #agent = DQNAgent(state_size, num_actions)
    Experience = namedtuple("Experience", field_names="state action reward next_state done")
    success_rate = 0.0
    success_rates = []
    for epoch in range(num_epochs):

        # Decay epsilon linearly from eps_max to eps_min
        eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)
        successes = 0
        for cycle in range(num_cycles):

            for episode in range(num_episodes):

                # Run episode and cache trajectory
                episode_trajectory = []
                state, goal = self.env.reset()

                for step in range(self.env.max_steps):

                    state_ = torch.cat((state, goal))
                    action = self.agent.take_action(state_, eps)
                    next_state, reward, done = self.env.step(action.item())
                    episode_trajectory.append(Experience(state, action, reward, next_state, done))
                    state = next_state
                    if done:
                        successes += 1
                        break

                # Fill up replay memory
                steps_taken = step
                for t in range(steps_taken):

                    # Standard experience replay
                    state, action, reward, next_state, done = episode_trajectory[t]
                    state_, next_state_ = torch.cat((state, goal)), torch.cat((next_state, goal))
                    self.agent.push_experience(state_, action, reward, next_state_, done)

                    # Hindsight experience replay
                    if HER:
                        for _ in range(future_k):
                            future = random.randint(t, steps_taken)  # index of future time step
                            new_goal = episode_trajectory[future].next_state  # take future next_state and set as goal
                            new_reward, new_done = self.env.compute_reward(next_state, new_goal)
                            state_, next_state_ = torch.cat((state, new_goal)), torch.cat((next_state, new_goal))
                            self.agent.push_experience(state_, action, new_reward, next_state_, new_done)

            # Optimize DQN
            for opt_step in range(num_opt_steps):
                self.agent.optimize_model()

            self.agent.update_target_network()

        success_rate = successes / (num_episodes * num_cycles)
        #print(success_rate)
        print("Epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * eps, success_rate))
        success_rates.append(success_rate)
    return success_rates
  
  def evaluate(self,num_tests):
    successes=0
    eps=0
    for trial in range(num_tests):
      state,goal=self.env.reset()
      for step in range(self.env.max_steps):
        state_ = torch.cat((state, goal))
        action = self.agent.take_action(state_, eps)
        next_state, reward, done = self.env.step(action.item())
        state = next_state
        if done:
          successes += 1
          break
    return ("Test success rate:"+str(successes/num_tests))