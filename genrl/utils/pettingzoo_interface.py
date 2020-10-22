from abc import ABC, abstractmethod

import gym
import numpy as np


class PettingZooInterface(ABC):
    """
    An interface between the PettingZoo API and agents defined in GenRL

    Attributes:

    env (PettingZoo Environment) : The environments in which the agents are acting
    agents_list (list) : A list containing all the agent objects present in the environment
    """

    def __init__(self, env, agents_list):
        self.env = env
        self.agents_list = agents_list

    def get_env_properties(self, network: str):
        state_dim = list(self.env.observation_spaces.values())[0].shape[0]
        if isinstance(list(self.env.action_spaces.vales())[0], gym.spaces.Discrete):
            discrete = True
            action_dim = list(self.env.action_spaces.values())[0].n
            action_lim = None
        elif isinstance(list(self.env.action_spaces.values())[0], gym.spaces.Box):
            discrete = False
            action_dim = list(self.env.action_spaces.values())[0].shape[0]
            action_lim = list(self.env.action_spaces.values())[0].high[0]
        else:
            NotImplementedError

        return state_dim, action_dim, discrete, action_lim

    def get_actions(self, states, steps, warmup_steps):
        if steps < warmup_steps:
            actions = {agent: self.env.action_spaces[agent].sample() for key in states}
        else:
            actions = {
                agent: self.agents_list[i].select_action(torch.tensor(states[agent]))
                for i, key in enumerate(states)
            }
        return actions

    def flatten(self, object):
        flattened_object = np.array([object[agent] for agent in self.env.agents])
        return flattened_object

    def trainer(self, action):
        raise NotImplementedError

    def update_agents(
        indiv_reward_batch_i,
        obs_batch_i,
        global_state_batch,
        global_actions_batch,
        global_next_state_batch,
        next_global_actions,
    ):
        raise NotImplementedError
