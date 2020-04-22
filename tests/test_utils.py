import pytest
import torch
import torch.nn as nn

from jigglypuffRL import MlpActorCritic, MlpPolicy, MlpValue
from jigglypuffRL.common.utils import *

class TestUtils:
    def test_get_model(self):
        ac = get_model('ac', 'mlp')
        p = get_model('p', 'mlp')
        v = get_model('v', 'mlp')
        
        assert ac == MlpActorCritic
        assert p == MlpPolicy
        assert v == MlpValue

    def test_mlp(self):
        sizes = [2,3,3,2]
        mlp_nn = mlp(sizes)
        mlp_nn_sac = mlp(sizes, sac=True)

        assert len(mlp_nn) == 2 * (len(sizes)- 1)
        assert all(isinstance(mlp_nn[i], nn.Linear) for i in range(0,5,2))
        assert len(mlp_nn_sac) == 2 * (len(sizes)- 2)
        assert all(isinstance(mlp_nn_sac[i], nn.Linear) for i in range(0,4,2))

        inp = torch.randn((2,))
        assert mlp_nn(inp).shape == (2,)
        assert mlp_nn_sac(inp).shape == (3,)