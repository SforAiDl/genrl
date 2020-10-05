import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from genrl.agents import ModelBasedAgent
from genrl.core import RolloutBuffer
from genrl.utils import get_env_properties, get_model, safe_mean


class CEM(ModelBasedAgent):
    def __init__(
        self,
        *args,
        network: str = "mlp",
        policy_layers: tuple = (100,),
        percentile: int = 70,
        **kwargs
    ):
        super(CEM, self).__init__(*args, **kwargs)
        self.network = network
        self.rollout_size = int(1e4)
        self.rollout = RolloutBuffer(self.rollout_size, self.env)
        self.policy_layers = policy_layers
        self.percentile = percentile

        self._create_model()
        self.empty_logs()

    def _create_model(self):
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
        self.optim = torch.optim.Adam(
            self.agent.parameters(), lr=1e-3
        )  # make this a hyperparam

    def plan(self, timesteps=1e4):
        state = self.env.reset()
        self.rollout.reset()
        _, _ = self.collect_rollouts(state)
        return (
            self.rollout.observations,
            self.rollout.actions,
            torch.sum(self.rollout.rewards).detach(),
        )

    def select_elites(self, states_batch, actions_batch, rewards_batch):
        reward_threshold = np.percentile(rewards_batch, self.percentile)
        print(reward_threshold)
        elite_states = [
            s.unsqueeze(0)
            for i in range(len(states_batch))
            if rewards_batch[i] >= reward_threshold
            for s in states_batch[i]
        ]
        elite_actions = [
            a.unsqueeze(0)
            for i in range(len(actions_batch))
            if rewards_batch[i] >= reward_threshold
            for a in actions_batch[i]
        ]

        return torch.cat(elite_states, dim=0), torch.cat(elite_actions, dim=0)

    def select_action(self, state):
        state = torch.as_tensor(state).float()
        action, dist = self.agent.get_action(state)
        return action

    def update_params(self):
        sess = [self.plan() for _ in range(100)]
        batch_states, batch_actions, batch_rewards = zip(*sess)
        elite_states, elite_actions = self.select_elites(
            batch_states, batch_actions, batch_rewards
        )
        print(elite_actions.shape)
        action_probs = self.agent.forward(torch.as_tensor(elite_states).float())
        print(action_probs.shape)
        print(self.action_dim)
        loss = F.cross_entropy(
            action_probs.view(-1, self.action_dim),
            torch.as_tensor(elite_actions).long().view(-1),
        )
        self.logs["crossentropy_loss"].append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
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
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            dones (:obj:`torch.Tensor`): Game over statuses of each environment
        """
        for i in range(self.rollout_size):
            action = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(action)

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                dones,
                torch.tensor(0),
                torch.tensor(0),
            )

            state = next_state

            self.collect_rewards(dones, i)

            if dones:
                break

        return torch.tensor(0), dones

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
                self.env.reset_single_env(i)

    def get_hyperparams(self):
        # return self.agent.get_hyperparams()
        pass

    def get_logging_params(self):
        logs = {
            "crossentropy_loss": safe_mean(self.logs["crossentropy_loss"]),
            "mean_reward": safe_mean(self.rewards),
        }
        return logs

    def empty_logs(self):
        # self.agent.empty_logs()
        self.logs = {}
        self.logs["crossentropy_loss"] = []
        self.rewards = []
