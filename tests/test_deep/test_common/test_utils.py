import pytest
import torch
import torch.nn as nn
import gym
import os
from shutil import rmtree

from genrl.deep.common import MlpActorCritic, MlpPolicy, MlpValue, CNNValue, venv
from genrl.deep.common.utils import *
from genrl import PPO1


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
        assert v_ == CNNValue

    def test_mlp(self):
        """
        test getting sequential MLP
        """
        sizes = [2, 3, 3, 2]
        mlp_nn = mlp(sizes)
        mlp_nn_sac = mlp(sizes, sac=True)

        assert len(mlp_nn) == 2 * (len(sizes) - 1)
        assert all(isinstance(mlp_nn[i], nn.Linear) for i in range(0, 5, 2))
        assert len(mlp_nn_sac) == 2 * (len(sizes) - 2)
        assert all(isinstance(mlp_nn_sac[i], nn.Linear) for i in range(0, 4, 2))

        inp = torch.randn((2,))
        assert mlp_nn(inp).shape == (2,)
        assert mlp_nn_sac(inp).shape == (3,)

    def test_cnn(self):
        """
        test getting CNN layers
        """
        channels = [1, 2, 4]
        kernels = [4, 1]
        strides = [2, 2]
        input_size = 84

        cnn_nn, output_size = cnn(channels, kernels, strides, input_size)

        assert len(cnn_nn) == 2 * (len(channels) - 1)
        assert all(isinstance(cnn_nn[i], nn.Conv2d) for i in range(0, len(channels), 2))
        assert all(
            isinstance(cnn_nn[i], nn.ReLU) for i in range(1, len(channels) + 1, 2)
        )
        assert output_size == 1764

    def test_save_params(self):
        """
        test saving algorithm state dict
        """
        env = venv("CartPole-v0", 1)
        algo = PPO1("mlp", env, epochs=1, save_model="test_ckpt")
        algo.learn()

        assert len(os.listdir("test_ckpt/PPO1_CartPole-v0")) != 0

    def test_load_params(self):
        """
        test loading algorithm parameters
        """
        env = venv("CartPole-v0", 1)
        algo = PPO1(
            "mlp", env, epochs=1, load_model="test_ckpt/PPO1_CartPole-v0/0-log-0.pt"
        )

        rmtree("test_ckpt")

    def test_get_env_properties(self):
        """
        test getting environment properties
        """
        env = venv("CartPole-v0", 1)

        state_dim, action_dim, discrete, _ = get_env_properties(env)
        assert state_dim == 4
        assert action_dim == 2
        assert discrete == True

        env = venv("Pendulum-v0", 1)

        state_dim, action_dim, discrete, action_lim = get_env_properties(env)
        assert state_dim == 3
        assert action_dim == 1
        assert discrete == False
        assert action_lim == 2.0

    def test_set_seeds(self):
        set_seeds(42)
        sampled = random.sample([i for i in range(20)], 1)[0]
        assert sampled == 3
