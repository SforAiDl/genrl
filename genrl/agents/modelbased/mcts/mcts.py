import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from genrl.agents.modelbased.mcts.base import TreePlanner, TreeSearchAgent, Node


class MCTSAgent(TreeSearchAgent):
    def __init__(self, *args, **kwargs):
        super(MCTSAgent, self).__init__(*args, **kwargs)
        self.planner = self._create_planner()
    
    def _create_planner(self):
        prior_policy = None
        rollout_policy = None
        return MCTSPlanner(prior_policy, rollout_policy)

    def _create_model(self):
        if isinstance(self.network, str):
            arch_type = self.network
            if self.shared_layers is not None:
                arch_type += "s"
            self.ac = get_model("ac", arch_type)(
                state_dim,
                action_dim,
                shared_layers=self.shared_layers,
                policy_layers=self.policy_layers,
                value_layers=self.value_layers,
                val_type="V",
                discrete=discrete,
                action_lim=action_lim,
            ).to(self.device)

        actor_params, critic_params = self.ac.get_params()
        self.optimizer_policy = opt.Adam(actor_params, lr=self.lr_policy)
        self.optimizer_value = opt.Adam(critic_params, lr=self.lr_value)

    def select_action(self, state) -> torch.Tensor:
        # return action.detach(), value, dist.log_prob.cpu()
        pass

    def get_traj_loss(self) -> None:
        pass

    def evaluate_actions(self, states, actions):
        # return values, dist.log_prob(actions).cpu(), dist.entropy.cpu()
        pass

    def update_params(self) -> None:
        # Blah blah
        # policy_loss = something
        # value_loss = something
        pass


class MCTSNode(Node):
    def __init__(self, *args, prior, **kwargs):
        super(MCTSNode, self).__init__(*args, **kwargs)
        self.value = 0
        self.prior = prior

    def selection_rule(self):
        if not self.children:
            return None
        actions = list(self.children.keys())
        counts = np.argmax([self.children[a] for a in actions])
        return actions[np.max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def sampling_rule(self):
        if self.children:
            actions = list(self.children.keys())
            idx = [self.children[a].selection_strategy(temp) for a in actions]
            return actions[]
        return None

    def expand(self, actions_dist):
        actions, probs = actions_dist
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = MCTSNode(parent=self, planner=self.planner, prior=probs[i])

    def _update(self, total_rew):
        self.count += 1
        self.value += 1 / self.count * (total_rew - self.value)

    def update_branch(self, total_rew):
        self._update(total_rew)
        if self.parent:
            self.parent.update_branch(total_rew)

    def get_child(self, action, obs=None):
        child = self.children[action]
        if obs is not None:
            if str(obs) not in child.children:
                child.children[str(obs)] = MCTSNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(obs)]
        return child
    
    def selection_strategy(self, temp=0):
        if not self.parent:
            return self.get_value()
        return self.get_value + temp * len(self.parent.children) * self.prior / (self.count-1)

    def get_value(self):
        return self.value

    def convert_visits_to_prior(self, reg=0.5):
        self.count = 0
        total_count = np.sum([(child.count + 1) for child in self.children])
        for child in self.children.values():
            child.prior = reg * (child.count + 1) / total_counts + reg / len(self.children)
            child.convert_visits_to_prior()


class MCTSPlanner(TreePlanner):
    def __init__(self, *args, prior, rollout_policy, episodes, **kwargs):
        super(MCTSPlanner, self).__init__(*args, **kwargs)
        self.env = env
        self.prior = prior
        self.rollout_policy = rollout_policy
        self.gamma = gamma
        self.episodes = episodes

    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)

    def _mc_search(self, state, obs):
        # Runs one iteration of mcts
        node = self.root
        total_rew = 0
        depth = 0
        terminal = False
        while depth < self.horizon and node.children and not terminal:
            action = node.sampling_rule()  # Not so sure about this
            obs, reward, terminal, _ = self.step(state, action)
            total_rew += self.gamma ** depth * reward
            node_obs = obs
            node = node.get_child(action, node_obs)
            depth += 1

        if not terminal:
            total_rew = self.eval(state, obs, total_rew, depth=depth)
        node.update_branch(total_rew)

    def eval(self, state, obs, total_rew=0, depth=0):
        # Run the rollout policy to yeild a sample for the value
        for h in range(depth, self.horizon):
            actions, probs = self.rollout_policy(state, obs)
            action = None  # rew Select an action
            obs, reward, terminal, _ = self.step(state, action)
            total_ += self.gamma ** h * reward
            if np.all(terminal):
                break
        return total_rew

    def plan(self, obs):
        for i in range(self.episodes):
            self._mc_search(copy.deepcopy(state), obs)
        return self.get_plan()

    def step_planner(seld, action):
        if self.step_strategy == "prior":
            self._step_by_prior(action)
        else:
            super().step_planner(action)
    
    def _step_by_prior(self, action):
        self._step_by_subtree(action)
        self.root.convert_visits_to_prior()
