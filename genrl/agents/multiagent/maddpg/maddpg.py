from copy import deepcopy
from typing import Any, Tuple

import numpy as np
import torch
import torch.optim as opt
from torch.nn import functional as F

from genrl.agents import DDPG
from genrl.core import MultiAgentReplayBuffer, ReplayBufferSamples
from genrl.utils import PettingZooInterface, get_model


class MADDPG(ABC):
    """MultiAgent Controller using the MADDPG algorithm

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        shared_layers(:obj:`tuple` of :obj:`int`): Sizes of shared layers in Actor Critic if using
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the critic
        replay_size (int): Capacity of the Replay Buffer
        polyak (float): Target model update parameter (1 for hard update)
        noise (:obj:`ActionNoise`): Action Noise function added to aid in exploration
        noise_std (float): Standard deviation of the action noise distribution
        max_ep_len (int): Maximum Episode length for training
        max_timesteps (int): Maximum limit of timesteps to train for
        warmup_steps (int): Number of warmup steps (random actions are taken to add randomness to the training)
        start_update (int): Timesteps after which the agent networks should start updating
        update_interval (int): Timesteps between target network updates
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self,
        network: Any,
        env: Any,
        batch_size: int = 64,
        gamma: float = 0.99,
        shared_layers=None,
        policy_layers: Tuple = (64, 64),
        value_layers: Tuple = (64, 64),
        lr_policy: float = 0.0001,
        lr_value: float = 0.001,
        replay_size: int = int(1e6),
        polyak: float = 0.995,
        noise: ActionNoise = None,
        noise_std: float = 0.2,
        max_ep_len: int = 200,
        max_timesteps: int = 5000,
        warmup_steps=1000,
        start_update: int = 1000,
        update_interval: int = 50,
        **kwargs,
    ):
        self.noise = noise
        self.doublecritic = False
        self.noise_std = noise_std
        self.gamma = self.gamma
        self.env = env
        self.network = network
        self.batch_size = batch_size
        self.lr_value = lr_value
        self.num_agents = self.env.num_agents
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, buffer_maxlen)
        self.render = render
        self.warmup_steps = warmup_steps
        self.shared_layers = shared_layers
        self.policy_layers = policy_layers
        self.value_layers = value_layers
        self.max_ep_len = max_ep_len
        self.max_timesteps = max_timesteps
        ac = self._create_model()
        self.agents = [
            DDPG(
                network=ac, env=env, lr_policy=lr_policy, lr_value=lr_value, gamma=gamma
            )
            for agent in self.env.agents
        ]
        self.EnvInterface = PettingZooInterface(self.env, self.agents)

    def _create_model(self):
        state_dim, action_dim, discrete, _ = self.EnvInterface.get_env_properties()
        if discrete:
            raise Exception(
                "Discrete Environments not supported for {}.".format(__class__.__name__)
            )

        if self.noise is not None:
            self.noise = self.noise(
                torch.zeros(action_dim), self.noise_std * torch.ones(action_dim)
            )

        if isinstance(self.network, str):
            arch_type = self.network
            arch_type += "c"
            if self.shared_layers is not None:
                raise NotImplementedError
            ac = get_model("ac", arch_type)(
                state_dim,
                action_dim,
                self.num_agents,
                self.shared_layers,
                self.policy_layers,
                self.value_layers,
                "Qsa",
                False,
            ).to(self.device)
        else:
            ac = self.network

        return ac

    def get_target_q_values(self, agent, global_batch, segmented_batch):
        global_next_actions = [
            agent.ac_target.get_action(
                segmented_batch[3][:, i, :], deterministic=True
            ).numpy()
            for agent, i in enumerate(self.agents)
        ]
        global_next_actions = torch.cat(global_next_actions, dim=1)
        global_next_actions = global_next_actions.float()

        if self.doublecritic:
            next_q_target_values = agent.ac_target.get_value(
                torch.cat([agent_batch.next_states, global_next_actions], dim=-1),
                mode="min",
            )
        else:
            next_q_target_values = agent.ac_target.get_value(
                torch.cat([agent_batch.next_states, global_next_actions], dim=-1)
            )

        target_q_values = (
            agent_batch.rewards
            + self.gamma * (1 - agent_batch.dones) * next_q_target_values
        )

        return target_q_values

    def get_q_loss(self, agent, agent_batch, segmented_batch):
        q_values = agent.get_q_values(global_batch.states, global_batch.actions)
        target_q_values = self.get_target_q_values(agent, agent_batch, segmented_batch)

        if self.doublecritic:
            loss = F.mse_loss(q_values[0], target_q_values) + F.mse_loss(
                q_values[1], target_q_values
            )
        else:
            loss = F.mse_loss(q_values, target_q_values)

        return loss

    def get_p_loss(self, agent, global_state_batch, segmented_states_batch):
        global_next_best_actions = [
            agent.ac.get_action(
                segmented_states_batch[:, i, :], deterministic=True
            ).numpy()
            for agent, i in enumerate(self.agents)
        ]
        global_next_best_actions = torch.cat(global_next_best_actions, dim=1)
        global_next_best_actions = global_next_best_actions.float()

        q_values = agent.ac.get_value(
            torch.cat([global_state_batch, global_next_best_actions], dim=-1)
        )
        policy_loss = -torch.mean(q_values)
        return policy_loss

    def update(self):
        segmented_batch, global_batch = self.replay_buffer.sample(self.batch_size)

        for transition in segmented_batch:
            for i, _ in enumerate(segmented_batch):
                transition[i] = self.EnvInterface.flatten(transition[i])

        (
            segmented_states,
            segmented_actions,
            segmented_rewards,
            segmented_next_states,
            segmented_dones,
        ) = map(np.stack, zip(*bitch))
        segmented_batch = [
            torch.from_numpy(v).float()
            for v in [
                segmented_states,
                segmented_actions,
                segmented_rewards,
                segmented_next_states,
                segmented_dones,
            ]
        ]

        for i, agent in enumerate(self.agents):
            agent_rewards_v = torch.reshape(global_batch[2][:, i], (self.batch_size, 1))
            agent_dones_v = torch.reshape(global_batch[4][:, i], (self.batch_size, 1))
            agent_batch_v = ReplayBufferSamples(
                *[
                    global_batch[0],
                    global_batch[1],
                    agent_rewards_v,
                    global_batch[3],
                    agent_dones_v,
                ]
            )
            value_loss = self.get_q_loss(
                agent=agent, agent_batch=agent_batch_v, segmented_batch=segmented_batch
            )

            value_loss.backward()
            agent.optimizer_value.step()

            agent_states_p = segmented_batch[0][:, i, :]
            policy_loss = self.get_p_loss(agent, global_batch[0], segmented_batch[0])

            policy_loss.backward()
            agent.optimizer_policy.step()

        for agent in self.agents:
            agent.update_target_model()

    def train(self):
        episode_rewards = []
        for episode in range(self.max_ep_len):
            states = self.env.reset()
            episode_reward = 0
            step = -1
            for step in range(self.max_timesteps):
                if self.render:
                    self.env.render(mode="human")

                step += 1
                actions = self.EnvInterface.get_actions(
                    states,
                    steps,
                    self.warmup_steps,
                    type="offpolicy",
                    deterministic=True,
                )
                next_states, rewards, dones, _ = self.env.step(actions)
                step_rewards = self.EnvInterface.flatten(rewards)
                episode_reward += np.mean(step_rewards)
                step_dones = self.EnvInterface.flatten(dones)
                if all(step_dones) or step == max_steps - 1:
                    dones = {agent: True for agent in self.env.agents}
                    self.replay_buffer.push(
                        [states, actions, rewards, next_states, dones]
                    )
                    episode_rewards.append(episode_reward)
                    print(
                        f"Episode: {episode + 1} | Steps Taken: {step +1} | Reward {episode_reward}"
                    )
                    break
                else:
                    dones = {agent: False for agent in self.env.agents}

                    self.replay_buffer.push(
                        [states, actions, rewards, next_states, dones]
                    )
                    states = next_states

                    if step >= self.start_update and step % self.update_interval == 0:
                        self.update()
