from jigglypuffRL.common.base import BaseValue
from jigglypuffRL.common.utils import mlp


def _get_val_model(
    arch, val_type, s_dim, hidden, a_dim=None,
):
    if val_type == "V":
        return arch([s_dim] + list(hidden) + [1])
    elif val_type == "Qsa":
        return arch([s_dim + a_dim] + list(hidden) + [1])
    elif val_type == "Qs":
        return arch([s_dim] + list(hidden) + [a_dim])


class MlpValue(BaseValue):
    """
    MLP Value Function
    :param s_dim: (int) state dimension of environment
    :param a_dim: (int) action dimension of environment
    :param val_type: (str) type of value function.
        'V' for V(s), 'Qs' for Q(s), 'Qsa' for Q(s,a)
    :param hidden: (tuple or list) sizes of hidden layers
    """

    def __init__(self, s_dim, a_dim=None, val_type="V", hidden=(32, 32)):
        super(MlpValue, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.model = _get_val_model(mlp, val_type, s_dim, hidden, a_dim)


v_registry = {"mlp": MlpValue}


def get_v_from_name(name_):
    if name_ in v_registry:
        return v_registry[name_]
    raise NotImplementedError
