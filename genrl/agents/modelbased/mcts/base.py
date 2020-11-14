import numpy as np

from genrl.agents.modelbased.base import ModelBasedAgent, Planner


class TreePlanner(Planner):
    def __init__(self):
        self.root = None
        self.observations = []
        self.horizon = horizon
        self.reset()

    def plan(self, state, obs):
        raise NotImplementedError()

    def get_plan(self):
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions

    def step(self, state, action):
        obs, reward, done, info = state.step(action)
        self.observations.append(obs)
        return obs, reward, done, info

    def step_tree(self, actions):
        if self.strategy == "reset":
            self.reset()
        elif self.strategy == "subtree":
            if actions:
                self._step_by_subtree(actions[0])
            else:
                self.reset()
        else:
            raise NotImplementedError

    def _step_by_subtree(self, action):
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.reset()

    def get_visits(self):
        visits = {}
        for obs in self.observations:
            if str(obs) not in visits.keys():
                visits[str(obs)] = 0
            visits[str(obs)] += 1

    def reset():
        raise NotImplementedError


class Node:
    def __init__(self, parent, planner):
        self.parent = parent
        self.planner = planner
        self.children = {}

        self.visits = 0

    def get_value(self):
        raise NotImplementedError

    def expand(self, branch_factor):
        self.children[a] = Node(self, planner)

    def selection_rule(self):
        raise NotImplementedError

    def is_leaf(self):
        return not self.children


class TreeSearchAgent(ModelBasedAgent):
    def __init__(self, *args, horizon, **kwargs):
        super(TreeSearchAgent, self).__init__(*args, **kwargs)
        self.planner = self._make_planner()
        self.prev_actions = []
        self.horizon = horizon
        self.remaining_horizon = 0
        self.steps = 0

    def _create_planner(self):
        pass

    def plan(self, obs):
        self.steps += 1
        replan = self._replan(self.prev_actions)
        if replan:
            env = self.env
            actions = self.planner.plan(state=env, obs=obs)
        else:
            actions = self.prev_actions[1:]

        self.prev_actions = actions
        return actions

    def _replan(self, actions):
        replan = self.remaining_horizon == 0 or len(actions) <= 1
        if replan:
            self.remaining_horizon = self.horizon
        else:
            self.remaining_horizon -= 1

        self.planner.step_tree(actions)
        return replan

    def reset(self):
        self.planner.reset()
        self.remaining_horizon = 0
        self.steps = 0
