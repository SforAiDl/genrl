import random

import gym
import torch
from torch import nn

from genrl.agents import PPO1
from genrl.core import CnnValue, MlpActorCritic, MlpPolicy, MlpValue
from genrl.environments import VectorEnv
from genrl.trainers import OnPolicyTrainer
from genrl.utils import (
    cnn,
    get_env_properties,
    get_model,
    mlp,
    mlp_concat,
    set_seeds,
    shared_mlp,
)


class TestUtils:
    def test_get_model(self):
        """
        test getting policy, value and AC models
        """
        ac = get_model("ac", "mlp")
        p = get_model("p", "mlp")
        v = get_model("v", "mlp")
        v_ = get_model("v", "cnn")

        assert ac == MlpActorCritic
        assert p == MlpPolicy
        assert v == MlpValue
        assert v_ == CnnValue

    def test_mlp(self):
        """
        test getting sequential MLP
        """
        sizes = [2, 3, 3, 2]
        mlp_nn = mlp(sizes)
        mlp_nn_sac = mlp(sizes, sac=True)
        mlp_nn_concat = mlp(sizes, concat_ind=1, sac=False)
        mlp_nn_concat_sac = mlp_concat(sizes, concat_ind=1, sac=True)
        shared_mlp_nn1, shared_mlp_nn2 = shared_mlp(
            sizes, sizes, sizes, sizes, sizes, sac=False
        )
        shared_mlp_nn1_sac, shared_mlp_nn2_sac = shared_mlp(
            sizes, sizes, sizes, sizes, sizes, sac=True
        )

        assert len(mlp_nn) == 2 * (len(sizes) - 1)
        assert all(isinstance(mlp_nn[i], nn.Linear) for i in range(0, 5, 2))
        assert len(mlp_nn_concat) == 2 * (len(sizes) - 1)
        assert all(isinstance(mlp_nn_concat[i], nn.Linear) for i in range(0, 5, 2))
        assert len(mlp_nn_sac) == 2 * (len(sizes) - 2)
        assert all(isinstance(mlp_nn_sac[i], nn.Linear) for i in range(0, 4, 2))
        assert len(mlp_nn_concat_sac) == 2 * (len(sizes) - 2)
        assert all(isinstance(mlp_nn_concat_sac[i], nn.Linear) for i in range(0, 4, 2))
        assert len(shared_mlp_nn1) == 2 * (len(sizes) - 1) * 3
        assert len(shared_mlp_nn2) == 2 * (len(sizes) - 1) * 3
        assert all(isinstance(shared_mlp_nn1[i], nn.Linear) for i in range(0, 8, 2))
        assert all(isinstance(shared_mlp_nn2[i], nn.Linear) for i in range(0, 8, 2))
        assert len(shared_mlp_nn1_sac) == 2 * (len(sizes) - 2) * 3
        assert all(isinstance(shared_mlp_nn1_sac[i], nn.Linear) for i in range(0, 4, 2))
        assert len(shared_mlp_nn2_sac) == 2 * (len(sizes) - 2) * 3
        assert all(isinstance(shared_mlp_nn2_sac[i], nn.Linear) for i in range(0, 4, 2))

        inp = torch.randn((2,))
        assert mlp_nn(inp).shape == (2,)
        assert mlp_nn_concat(inp).shape == (2,)
        assert shared_mlp_nn1(inp).shape == (2,)
        assert shared_mlp_nn2(inp).shape == (2,)
        assert mlp_nn_sac(inp).shape == (3,)
        assert mlp_nn_concat_sac(inp).shape == (3,)
        assert shared_mlp_nn1_sac(inp).shape == (3,)
        assert shared_mlp_nn2_sac(inp).shape == (3,)

    def test_cnn(self):
        """
        test getting CNN layers
        """
        channels = [1, 2, 4]
        kernels = [4, 1]
        strides = [2, 2]

        cnn_nn, output_size = cnn(channels, kernels, strides)

        assert len(cnn_nn) == 2 * (len(channels) - 1)
        assert all(isinstance(cnn_nn[i], nn.Conv2d) for i in range(0, len(channels), 2))
        assert all(
            isinstance(cnn_nn[i], nn.ReLU) for i in range(1, len(channels) + 1, 2)
        )
        assert output_size == 1764

    def test_get_env_properties(self):
        """
        test getting environment properties
        """
        env = VectorEnv("CartPole-v0", 1)

        state_dim, action_dim, discrete, _ = get_env_properties(env)
        assert state_dim == 4
        assert action_dim == 2
        assert discrete is True

        env = VectorEnv("Pendulum-v0", 1)

        state_dim, action_dim, discrete, action_lim = get_env_properties(env)
        assert state_dim == 3
        assert action_dim == 1
        assert discrete is False
        assert action_lim == 2.0

    def test_set_seeds(self):
        set_seeds(42)
        sampled = random.sample([i for i in range(20)], 1)[0]
        assert sampled == 3
