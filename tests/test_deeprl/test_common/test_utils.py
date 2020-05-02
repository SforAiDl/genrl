import pytest
import torch
import torch.nn as nn
import gym
import os
from shutil import rmtree

from jigglypuffRL import (
    MlpActorCritic,
    MlpPolicy,
    MlpValue,
    PPO1,
)
from jigglypuffRL.common.utils import *


class TestUtils:
    def test_get_model(self):
        """
        test getting policy, value and AC models
        """
        ac = get_model("ac", "mlp")
        p = get_model("p", "mlp")
        v = get_model("v", "mlp")

        assert ac == MlpActorCritic
        assert p == MlpPolicy
        assert v == MlpValue

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

    def test_evaluate(self):
        """
        test evaluating trained algorithm
        """
        env = gym.make("CartPole-v0")
        algo = PPO1("mlp", env, epochs=1)
        algo.learn()
        evaluate(algo, num_timesteps=10)

    def test_save_params(self):
        """
        test saving algorithm state dict
        """
        env = gym.make("CartPole-v0")
        algo = PPO1("mlp", env, epochs=1, save_model="test_ckpt")
        algo.learn()

        assert len(os.listdir("test_ckpt/PPO1_CartPole-v0")) != 0

    def test_load_params(self):
        """
        test loading algorithm parameters
        """
        env = gym.make("CartPole-v0")
        algo = PPO1(
            "mlp", env, epochs=1, pretrained="test_ckpt/PPO1_CartPole-v0/0-log-0.pt"
        )

        rmtree("test_ckpt")
