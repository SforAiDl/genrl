from abc import ABC

import numpy as np
import torch


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


class ModelBasedAgent(ABC):
    def __init__(self, env, planner=None, render=False, device="cpu"):
        self.env = env
        self.planner = planner
        self.render = render
        self.device = torch.device(device)

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

    def update_params(self):
        """
        Update the parameters (Parameters of the learnt model and/or Parameters of the policy being used)
        """
        raise NotImplementedError

    def get_hyperparans(self):
        raise NotImplementedError

    def get_logging_params(self):
        raise NotImplementedError

    def _load_weights(self, weights):
        raise NotImplementedError

    def empty_logs(self):
        raise NotImplementedError
