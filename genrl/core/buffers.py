import copy
import random
from collections import deque
from typing import NamedTuple, Tuple

import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class ReplayBuffer:
    """
    Implements the basic Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :type capacity: int
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, inp: Tuple) -> None:
        """
        Adds new experience to buffer

        :param inp: Tuple containing state, action, reward, next_state and done
        :type inp: tuple
        :returns: None
        """
        self.memory.append(inp)

    def sample(
        self, batch_size: int
    ) -> (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
                Returns randomly sampled experiences from replay memory

                :param batch_size: Number of samples per batch
                :type batch_size: int
                :returns: (Tuple composing of `state`, `action`, `reward`,
        `next_state` and `done`)
        """
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return [
            torch.from_numpy(v).float()
            for v in [state, action, reward, next_state, done]
        ]

    def __len__(self) -> int:
        """
        Gives number of experiences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.memory)


class PrioritizedBuffer:
    """
    Implements the Prioritized Experience Replay Mechanism

    :param capacity: Size of the replay buffer
    :param alpha: Level of prioritization
    :type capacity: int
    :type alpha: int
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)

    def push(self, inp: Tuple) -> None:
        """
                Adds new experience to buffer

                :param inp: (Tuple containing `state`, `action`, `reward`,
        `next_state` and `done`)
                :type inp: tuple
                :returns: None
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(inp)
        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int, beta: float = None
    ) -> (
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """
                (Returns randomly sampled memories from replay memory along with their
        respective indices and weights)

                :param batch_size: Number of samples per batch
                :param beta: (Bias exponent used to correct
        Importance Sampling (IS) weights)
                :type batch_size: int
                :type beta: float
                :returns: (Tuple containing `states`, `actions`, `next_states`,
        `rewards`, `dones`, `indices` and `weights`)
        """
        if beta is None:
            beta = self.beta

        total = len(self.buffer)
        priorities = np.asarray(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(total, batch_size, p=probabilities)

        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.asarray(weights, dtype=np.float32)

        samples = [self.buffer[i] for i in indices]
        (states, actions, rewards, next_states, dones) = map(np.stack, zip(*samples))

        return [
            torch.as_tensor(v, dtype=torch.float32)
            for v in [
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ]
        ]

    def update_priorities(self, batch_indices: Tuple, batch_priorities: Tuple) -> None:
        """
                Updates list of priorities with new order of priorities

                :param batch_indices: List of indices of batch
                :param batch_priorities: (List of priorities of the batch at the
        specific indices)
                :type batch_indices: list or tuple
                :type batch_priorities: list or tuple
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[int(idx)] = priority.mean()

    def __len__(self) -> int:
        """
        Gives number of expesampleriences in buffer currently

        :returns: Length of replay memory
        """
        return len(self.buffer)

    @property
    def pos(self):
        return len(self.buffer)


class HERWrapper:
    """
    A wrapper class to convert a replay buffer to a HER Style Buffer

    Args:
        replay_buffer (ReplayBuffer): An instance of the replay buffer to be converted to a HER style buffer
        n_sampled_goals (int): The number of artificial transitions to generate for each actual transition
        goal_selection_strategy (str): The strategy to be used to generate goals for the artificial transitions
        env (HerGoalEnvWrapper): The goal env, wrapped using HERGoalEnvWrapper
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, env):

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.replay_buffer = replay_buffer
        self.transitions = []
        self.allowed_strategies = ["future", "final", "episode", "random"]
        self.env = env

    def push(self, inp: Tuple):
        state, action, reward, next_state, done, info = inp
        if isinstance(state, dict):
            state = self.env.convert_dict_to_obs(state)
            next_state = self.env.convert_dict_to_obs(next_state)

        self.transitions.append((state, action, reward, next_state, done, info))
        self.replay_buffer.push((state, action, reward, next_state, done))

        if inp[-1]:
            self._store_episode()
            self.transitions = []

    def sample(self, batch_size):
        return self.replay_buffer.sample(batch_size)

    def _sample_achieved_goal(self, ep_transitions, transition_idx):
        if self.goal_selection_strategy == "future":
            # Sample a goal that was observed in the future
            selected_idx = np.random.choice(
                np.arange(transition_idx + 1, len(ep_transitions))
            )
            selected_transition = ep_transitions[selected_idx]
        elif self.goal_selection_strategy == "final":
            # Sample the goal that was finally achieved during the episode
            selected_transition = ep_transitions[-1]
        elif self.goal_selection_strategy == "episode":
            # Sample a goal that was observed in the episode
            selected_idx = np.random.choice(np.arange(len(ep_transitions)))
            selected_transition = ep_transitions[selected_idx]
        elif self.goal_selection_strategy == "random":
            # Sample a random goal from the entire replay buffer
            selected_idx = np.random.choice(len(self.replay_buffer))
            selected_transition = self.replay_buffer.memory[selected_idx]
        else:
            raise ValueError(
                f"Goal selection strategy must be one of {self.allowed_strategies}"
            )

        return self.env.convert_obs_to_dict(selected_transition[0])["achieved_goal"]

    def _sample_batch_goals(self, ep_transitions, transition_idx):
        return [
            self._sample_achieved_goal(ep_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        for transition_idx, transition in enumerate(self.transitions):

            # We cannot sample from the future on the last step
            if (
                transition_idx == len(self.transitions) - 1
                and self.goal_selection_strategy == "future"
            ):
                break

            sampled_goals = self._sample_batch_goals(self.transitions, transition_idx)

            for goal in sampled_goals:
                state, action, reward, next_state, done, info = copy.deepcopy(
                    transition
                )

                # Convert concatenated obs to dict, so we can update the goals
                state_dict = self.env.convert_obs_to_dict(state)
                next_state_dict = self.env.convert_obs_to_dict(next_state)

                # Update the desired goals in the transition
                state_dict["desired_goal"] = goal
                next_state_dict["desired_goal"] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(
                    next_state_dict["achieved_goal"], goal, info
                )

                # Store the newly created transition in the replay buffer
                state = self.env.convert_dict_to_obs(state_dict)
                next_state = self.env.convert_dict_to_obs(next_state_dict)
                self.replay_buffer.push((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.replay_buffer)
