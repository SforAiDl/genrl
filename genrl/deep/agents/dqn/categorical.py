from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import optim as opt

from genrl.deep.agents.dqn.base import BaseDQN
from genrl.deep.agents.dqn.utils import get_projection_distribution
from genrl.deep.common import get_env_properties, get_model


class CategoricalDQN(BaseDQN):
    def __init__(
        self,
        *args,
        noisy_layers: Tuple = (32, 128),
        num_atoms: int = 51,
        v_min: int = -10,
        v_max: int = 10,
        **kwargs
    ):
        super(CategoricalDQN, self).__init__(*args, **kwargs)
        self.noisy_layers = noisy_layers
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.empty_logs()
        self.create_model()

    def create_model(self, *args) -> None:
        input_dim, action_dim, _, _ = get_env_properties(self.env, self.network_type)

        self.model = get_model("dv", self.network_type + "categorical")(
            input_dim, action_dim, self.layers, self.noisy_layers, self.num_atoms
        )
        self.target_model = deepcopy(self.model)

        self.replay_buffer = self.buffer_class(self.replay_size, *args)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr_value)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:

        if not deterministic:
            if np.random.rand() < self.epsilon:
                return np.asarray(self.env.sample())

        state = torch.FloatTensor(state)
        dist = self.model(state).data.cpu()
        dist = dist * torch.linspace(self.v_min, self.v_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()
        return action

    def get_q_loss(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.reshape(-1, *self.env.obs_shape)
        actions = actions.reshape(-1, *self.env.action_shape).long()
        next_states = next_states.reshape(-1, *self.env.obs_shape)

        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        projection_distribution = get_projection_distribution(
            next_states, rewards, dones, self.num_atoms, self.v_min, self.v_max
        )
        dist = self.model(states)
        actions = actions.unsqueeze(1).expand(-1, 1, self.num_atoms)
        dist = dist.gather(1, actions).squeeze(1)
        dist.data.clamp_(0.01, 0.99)

        loss = -(projection_distribution * dist.log()).sum(1).mean()
        self.logs["value_loss"].append(loss.item())
        return loss
