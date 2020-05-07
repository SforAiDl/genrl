from genrl.deep.common.base import BaseValue
from genrl.deep.common.utils import mlp


def _get_val_model(
    arch, val_type, state_dim, hidden, action_dim=None,
):
    """
    Returns Neural Network given specifications

    Args:
        :param arch: (str) Specifies type of architecture
            "mlp" for MLP layers
        :param val_type: (str) Specifies type of value function
            'V' for V(s), 'Qs' for Q(s), 'Qsa' for Q(s,a)
        :param state_dim: (int) State dimensions of environment
        :param action_dim: (int) Action dimensions of environment
        :param hidden: (tuple or list) Sizes of hidden layers
    """
    if val_type == "V":
        return arch([state_dim] + list(hidden) + [1])
    elif val_type == "Qsa":
        return arch([state_dim + action_dim] + list(hidden) + [1])
    elif val_type == "Qs":
        return arch([state_dim] + list(hidden) + [action_dim])
    else:
        raise ValueError


class MlpValue(BaseValue):
    """
    MLP Value Function
        :param state_dim: (int) State dimensions of environment
        :param action_dim: (int) Action dimensions of environment
        :param val_type: (str) Specifies type of value function
            "V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
        :param hidden: (tuple or list) Sizes of hidden layers
    """

    def __init__(self, state_dim, action_dim=None, val_type="V", hidden=(32, 32)):
        super(MlpValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = _get_val_model(mlp, val_type, state_dim, hidden, action_dim)


value_registry = {"mlp": MlpValue}


def get_value_from_name(name_):
    """
    Returns value function given the name of the value function

    Args:
        :param name_: (string) Name of the value function needed
    """
    if name_ in value_registry:
        return value_registry[name_]
    raise NotImplementedError
