from genrl.deep.common.base import BaseValue
from genrl.deep.common.utils import mlp, cnn


def _get_val_model(
    arch, val_type, state_dim, hidden, action_dim=None,
):
    """
    Returns Neural Network given specifications

    :param arch: Specifies type of architecture "mlp" for MLP layers
    :param val_type: Specifies type of value function: \
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
    :param state_dim: State dimensions of environment
    :param action_dim: Action dimensions of environment
    :param hidden: Sizes of hidden layers
    :type arch: str
    :type val_type: str
    :type state_dim: str
    :type action_dim: int
    :type hidden: tuple or list

    :returns: Neural Network model to be used for the Value function
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
    MLP Value Function class

    :param state_dim: State dimensions of environment
    :param action_dim: Action dimensions of environment
    :param val_type: Specifies type of value function: \
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
    :param hidden: Sizes of hidden layers
    :type state_dim: int
    :type action_dim: int
    :type val_type: str
    :type hidden: tuple or list
    """

    def __init__(
        self,
        state_dim,
        action_dim=None,
        val_type="V",
        hidden=(32, 32)
    ):
        super(MlpValue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = _get_val_model(
            mlp, val_type, state_dim, hidden, action_dim
        )


class CNNValue(BaseValue):
    """
    CNN Value Function class

    :param state_dim: State dimension of environment
    :param action_dim: Action dimension of environment
    :param history_length: Length of history of states
    :param val_type: Specifies type of value function: \
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
    :param hidden: Sizes of hidden layers
    :type state_dim: int
    :type action_dim: int
    :type history_length: int
    :type val_type: str
    :type hidden: tuple or list
    """
    def __init__(
        self, action_dim, history_length=4, val_type="Qs",
        fc_layers=(256,)
    ):
        super(CNNValue, self).__init__()

        self.action_dim = action_dim

        self.conv, output_size = cnn((history_length, 16, 32))

        self.fc = _get_val_model(
            mlp, val_type, output_size, fc_layers, action_dim
        )

    def forward(self, state):
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        state = self.fc(state)
        return state


value_registry = {"mlp": MlpValue, "cnn": CNNValue}


def get_value_from_name(value_name):
    """
    Gets the value function given the name of the value function

    :param value_name: Name of the value function needed
    :type value_name: str
    :returns: Value function
    """
    if value_name in value_registry:
        return value_registry[value_name]
    raise NotImplementedError
