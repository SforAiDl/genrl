from .base import BasePolicy
from .utils import mlp


class MlpPolicy(BasePolicy):
    """
    MLP Policy

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type discrete: bool
    """

    def __init__(
        self, state_dim, action_dim, hidden=(32, 32), disc=True, *args, **kwargs
    ):
        super(MlpPolicy, self).__init__(state_dim, action_dim, hidden, disc, **kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = mlp([state_dim] + list(hidden) + [action_dim], sac=self.sac)


policy_registry = {"mlp": MlpPolicy}


def get_policy_from_name(name_):
    """
    Returns policy given the name of the policy

    :param name_: Name of the policy needed
    :type name_: str
    :returns: Policy Function to be used
    """
    if name_ in policy_registry:
        return policy_registry[name_]
    raise NotImplementedError
