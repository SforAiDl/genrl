from genrl.deep.common.base import BasePolicy
from genrl.deep.common.utils import mlp


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
        self,
        state_dim,
        action_dim,
        hidden=(32, 32),
        discrete=True,
        *args,
        **kwargs
    ):
        super(MlpPolicy, self).__init__(
            state_dim, action_dim, hidden, discrete, **kwargs
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = mlp(
            [state_dim] + list(hidden) + [action_dim], sac=self.sac
        )


policy_registry = {"mlp": MlpPolicy}


def get_policy_from_name(policy_name):
    """
    Returns policy given the name of the policy

    :param policy_name: Name of the policy needed
    :type policy_name: str
    :returns: Policy Function to be used
    """
    if policy_name in policy_registry:
        return policy_registry[policy_name]
    raise NotImplementedError
