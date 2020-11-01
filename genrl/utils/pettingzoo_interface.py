from abc import ABC
from typing import Any, Dict, Tuple

import gym
import numpy as np


class PettingZooInterface(ABC):
    """
    An interface between the PettingZoo API and agents defined in GenRL

    Attributes:

    env (PettingZoo Environment) : The environments in which the agents are acting
    agents_list (list) : A list containing all the agent objects present in the environment
    """

    def __init__(self, env: Any, agents_list: list):
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
            raise NotImplementedError

        return state_dim, action_dim, discrete, action_lim

    def select_offpolicy_action(
        self, state: np.ndarray, agent, noise, deterministic: bool = False
    ):
        action, _ = agent.ac.get_action(torch.tensor(state), deterministic)
        action = action.detach()

        if noise is not None:
            action += noise

        return torch.clamp(
            action,
            list(self.env.action_spaces.values())[0].low[0],
            list(self.env.action_spaces.values())[0].high[0],
        ).numpy()

    def select_onpolicy_action(
        self, state: np.ndarray, agent, deterministic: bool = False
    ):
        raise NotImplementedError

    def get_actions(
        self,
        states: Dict[str, np.ndarray],
        steps: int,
        warmup_steps: int,
        type: str,
        deterministic: bool = False,
    ):
        if steps < warmup_steps:
            actions = {agent: self.env.action_spaces[agent].sample() for key in states}
        else:
            if type == "offpolicy":
                actions = {
                    agent: self.select_offpolicy_action(
                        states[agent], self.agents_list[i], deterministic, noise
                    )
                    for i, agent in enumerate(states)
                }
            elif type == "onpolicy":
                raise NotImplementedError
            else:
                raise NotImplementedError

        return actions

    def flatten(self, obj: Dict):
        flattened_object = np.array([obj[agent] for agent in self.env.agents])

        return flattened_object
