from typing import Generator, NamedTuple, Optional, Union

import gym
import torch

from genrl.core.buffer import BaseBuffer
from genrl.environments.vec_env import VecEnv


class RolloutBufferSamples(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer(BaseBuffer):
    """Rollout buffer used in on-policy algorithms like A2C/PPO

    buffer_size (int): Max number of element in the buffer
    device (:obj:`torch.device` or str):  PyTorch device to which the values will be converted
    env (Environment): The environment being trained on
    gae_lambda (float): Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    """

    def __init__(
        self,
        *args,
        env: Union[gym.Env, VecEnv],
        gae_lambda: float = 1,
        **kwargs,
    ):
        super(RolloutBuffer, self).__init__(*args, **kwargs)
        self.gae_lambda = gae_lambda
        self.observations, self.actions, self.rewards, self.advantages = (
            None,
            None,
            None,
            None,
        )
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros(
            *(self.buffer_size, self.env.n_envs, *self.env.obs_shape)
        )
        self.actions = torch.zeros(
            *(self.buffer_size, self.env.n_envs, *self.env.action_shape)
        )
        self.rewards = torch.zeros(self.buffer_size, self.env.n_envs)
        self.returns = torch.zeros(self.buffer_size, self.env.n_envs)
        self.dones = torch.zeros(self.buffer_size, self.env.n_envs)
        self.values = torch.zeros(self.buffer_size, self.env.n_envs)
        self.log_probs = torch.zeros(self.buffer_size, self.env.n_envs)
        self.advantages = torch.zeros(self.buffer_size, self.env.n_envs)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: torch.zeros,
        action: torch.zeros,
        reward: torch.zeros,
        done: torch.zeros,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: (torch.zeros) Observation
        :param action: (torch.zeros) Action
        :param reward: (torch.zeros)
        :param done: (torch.zeros) End of episode signal.
        :param value: (torch.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (torch.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = obs.detach().clone()
        self.actions[self.pos] = action.squeeze().detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.dones[self.pos] = done.detach().clone()
        self.values[self.pos] = value.detach().clone().flatten()
        self.log_probs[self.pos] = log_prob.detach().clone().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = torch.randperm(self.buffer_size * self.env.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.env.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.env.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
