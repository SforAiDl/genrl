from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from genrl.agents import ModelBasedAgent
from genrl.core import RolloutBuffer
from genrl.utils import get_env_properties, get_model, safe_mean


class CEM(ModelBasedAgent):
    """Cross Entropy method algorithm (CEM)

    Attributes:
        network (str): The type of network to be used
        env (Environment): The environment the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        policy_layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network of the policy
        lr_policy (float): learning rate of the policy
        percentile (float): Top percentile of rewards to consider as elite
        simulations_per_epoch (int): Number of simulations to perform before taking a gradient step
        rollout_size (int): Capacity of the replay buffer
        render (bool): Whether to render the environment or not
        device  (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        *args,
        network: str = "mlp",
        percentile: float = 70,
        simulations_per_epoch: int = 1000,
        rollout_size,
        **kwargs
    ):
        super(CEM, self).__init__(*args, **kwargs)
        self.network = network
        self.rollout_size = rollout_size
        self.rollout = RolloutBuffer(self.rollout_size, self.env)
        self.percentile = percentile
        self.simulations_per_epoch = simulations_per_epoch

        self._create_model()
        self.empty_logs()

    def _create_model(self):
        """Function to initialize the Policy

        This will create the Policy net for the CEM agent
        """
        self.state_dim, self.action_dim, discrete, action_lim = get_env_properties(
            self.env, self.network
        )
        self.agent = get_model("p", self.network)(
            self.state_dim,
            self.action_dim,
            self.policy_layers,
            "V",
            discrete,
            action_lim,
        )
        self.optim = torch.optim.Adam(self.agent.parameters(), lr=self.lr_policy)

    def plan(self):
        """Function to plan out one episode

        Returns:
            states (:obj:`list` of :obj:`torch.Tensor`): Batch of states the agent encountered in the episode
            actions (:obj:`list` of :obj:`torch.Tensor`): Batch of actions the agent took in the episode
            rewards (:obj:`torch.Tensor`): The episode reward obtained
        """
        state = self.env.reset()
        self.rollout.reset()
        states, actions = self.collect_rollouts(state)
        return (states, actions, self.rewards[-1])

    def select_elites(self, states_batch, actions_batch, rewards_batch):
        """Function to select the elite states and elite actions based on the episode reward

        Args:
            states_batch (:obj:`list` of :obj:`torch.Tensor`): Batch of states
            actions_batch (:obj:`list` of :obj:`torch.Tensor`): Batch of actions
            rewards_batch (:obj:`list` of :obj:`torch.Tensor`): Batch of rewards

        Returns:
            elite_states (:obj:`torch.Tensor`): Elite batch of states based on episode reward
            elite_actions (:obj:`torch.Tensor`): Actions the agent took during the elite batch of states

        """
        reward_threshold = np.percentile(rewards_batch, self.percentile)
        elite_states = torch.cat(
            [
                s.unsqueeze(0).clone()
                for i in range(len(states_batch))
                if rewards_batch[i] >= reward_threshold
                for s in states_batch[i]
            ],
            dim=0,
        )
        elite_actions = torch.cat(
            [
                a.unsqueeze(0).clone()
                for i in range(len(actions_batch))
                if rewards_batch[i] >= reward_threshold
                for a in actions_batch[i]
            ],
            dim=0,
        )

        return elite_states, elite_actions

    def select_action(self, state):
        """Select action given state

        Action selection policy for the Cross Entropy agent

        Args:
            state (:obj:`torch.Tensor`): Current state of the agent

        Returns:
            action (:obj:`torch.Tensor`): Action taken by the agent
        """
        state = torch.as_tensor(state).float()
        action, dist = self.agent.get_action(state)
        return action

    def update_params(self):
        """Updates the the Policy network of the CEM agent

        Function to update the policy network
        """
        sess = [self.plan() for _ in range(self.simulations_per_epoch)]
        batch_states, batch_actions, batch_rewards = zip(*sess)
        elite_states, elite_actions = self.select_elites(
            batch_states, batch_actions, batch_rewards
        )
        action_probs = self.agent.forward(elite_states.float())
        loss = F.cross_entropy(
            action_probs.view(-1, self.action_dim),
            elite_actions.long().view(-1),
        )
        self.logs["crossentropy_loss"].append(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optim.step()

    def get_traj_loss(self, values, dones):
        # No need for this here
        pass

    def collect_rollouts(self, state: torch.Tensor):
        """Function to collect rollouts

        Collects rollouts by playing the env like a human agent and inputs information into
        the rollout buffer.

        Args:
            state (:obj:`torch.Tensor`): The starting state of the environment

        Returns:
            states (:obj:`list`): list of states the agent encountered during the episode
            actions (:obj:`list`): list of actions the agent took in the corresponding states
        """
        states = []
        actions = []
        for i in range(self.rollout_size):
            action = self.select_action(state)
            states.append(state)
            actions.append(action)

            next_state, reward, dones, _ = self.env.step(action)

            if self.render:
                self.env.render()

            state = next_state

            self.collect_rewards(dones, i)

            if torch.any(dones.byte()):
                break

        return states, actions

    def collect_rewards(self, dones: torch.Tensor, timestep: int):
        """Helper function to collect rewards

        Runs through all the envs and collects rewards accumulated during rollouts

        Args:
            dones (:obj:`torch.Tensor`): Game over statuses of each environment
            timestep (int): Timestep during rollout
        """
        for i, done in enumerate(dones):
            if done or timestep == self.rollout_size - 1:
                self.rewards.append(self.env.episode_reward[i].detach().clone())

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get relevant hyperparameters to save

        Returns:
            hyperparams (:obj:`dict`): Hyperparameters to be saved
            weights (:obj:`torch.Tensor`): Neural network weights
        """
        hyperparams = {
            "network": self.network,
            "lr_policy": self.lr_policy,
            "rollout_size": self.rollout_size,
        }
        return hyperparams, self.agent.state_dict()

    def _load_weights(self, weights) -> None:
        self.agent.load_state_dict(weights)

    def get_logging_params(self) -> Dict[str, Any]:
        """Gets relevant parameters for logging

        Returns:
            logs (:obj:`dict`): Logging parameters for monitoring training
        """
        logs = {
            "crossentropy_loss": safe_mean(self.logs["crossentropy_loss"]),
            "mean_reward": safe_mean(self.rewards),
        }

        self.empty_logs()
        return logs

    def empty_logs(self):
        """Empties logs"""
        self.logs = {}
        self.logs["crossentropy_loss"] = []
        self.rewards = []
