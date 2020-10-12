import gym
import numpy as np
import pytest

from genrl.utils.models import TabularModel


class TestClassicalModel:
    def test_tabular_model(self):
        model = TabularModel(4, 2)

        assert model.s_model.shape == (4, 2)
        assert model.r_model.shape == (4, 2)
        assert model.is_empty() is True

        model.add(3, 1, 3.1, 1)
        assert model.is_empty() is False

        assert model.s_model[3, 1] == 1
        assert model.r_model[3, 1] == 3.1

        assert model.sample() == (3, 1)
        assert model.step(3, 1) == (3.1, 1)
