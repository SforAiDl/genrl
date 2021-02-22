from abc import ABC

import torch

from genrl.agents import BaseAgent


class Planner:
    def __init__(self, initial_state, dynamics_model=None):
        if dynamics_model is not None:
            self.dynamics_model = dynamics_model
        self.initial_state = initial_state

    def _learn_dynamics_model(self, state):
        raise NotImplementedError

    def plan(self):
        raise NotImplementedError

    def execute_actions(self):
        raise NotImplementedError


class ModelBasedAgent(BaseAgent):
    def __init__(self, *args, planner=None, **kwargs):
        super(ModelBasedAgent, self).__init__(*args, **kwargs)
        self.planner = planner

    def plan(self):
        """
        To be used to plan out a sequence of actions
        """
        if self.planner is not None:
            raise ValueError("Provide a planner to plan for the environment")
        self.planner.plan()

    def generate_data(self):
        """
        To be used to generate synthetic data via a model (may be learnt or specified beforehand)
        """
        raise NotImplementedError

    def value_equivalence(self, state_space):
        """
        To be used for approximate value estimation methods e.g. Value Iteration Networks
        """
        raise NotImplementedError
