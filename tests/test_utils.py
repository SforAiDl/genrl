import pytest
import torch.nn 

from jigglypuffRL import MlpActorCritic, MlpPolicy, MlpValue
from jigglypuffRL.common.utils import *

class TestUtils:
    def test_get_model(self):
        ac = get_model('ac', 'mlp')
        p = get_model('p', 'mlp')
        v = get_model('v', 'mlp')
        
        assert isinstance(ac(2,2), MlpActorCritic)
        assert isinstance(p(2,2), MlpPolicy)
        assert isinstance(v(2,2), MlpValue)
