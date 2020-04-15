from jigglypuffRL.common.base import BasePolicy
from jigglypuffRL.common.utils import mlp


class MlpPolicy(BasePolicy):
    """
    MLP Policy
    :param state_dim: (int) state dimension of environment
    :param action_dim: (int) action dimension of environment
    :param hidden: (tuple or list) sizes of hidden layers
    :param disc: (bool) discrete action space?
    :param det: (bool) deterministic policy?
    """

    def __init__(
        self, state_dim, action_dim, hidden=(32, 32), disc=True,
        *args, **kwargs
    ):
        super(MlpPolicy, self).__init__(
            disc, state_dim, action_dim, hidden, **kwargs
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = mlp(
            [state_dim] + list(hidden) + [action_dim], sac=self.sac
        )


policy_registry = {"mlp": MlpPolicy}


def get_policy_from_name(name_):
    if name_ in policy_registry:
        return policy_registry[name_]
    raise NotImplementedError
