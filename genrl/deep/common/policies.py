from genrl.deep.common.base import BasePolicy
from genrl.deep.common.utils import mlp


class MlpPolicy(BasePolicy):
    """
    MLP Policy
        :param state_dim: (int) State dimensions of the environment
        :param action_dim: (int) Action dimensions of the environment
        :param hidden: (tuple or list) Sizes of hidden layers
        :param disc: (bool) True if action space is discrete, else False
        :param det: (bool) True if policy is deterministic, else False
    """

    def __init__(
        self, state_dim, action_dim, hidden=(32, 32), disc=True, *args, **kwargs
    ):
        super(MlpPolicy, self).__init__(disc, state_dim, action_dim, hidden, **kwargs)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = mlp([state_dim] + list(hidden) + [action_dim], sac=self.sac)


policy_registry = {"mlp": MlpPolicy}


def get_policy_from_name(name_):
    """
    Returns policy given the name of the policy

    Args:
        :param name_: (string) Name of the policy needed
    """
    if name_ in policy_registry:
        return policy_registry[name_]
    raise NotImplementedError
