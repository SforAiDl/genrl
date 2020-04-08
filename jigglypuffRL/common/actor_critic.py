from jigglypuffRL.common.base import BaseActorCritic
from jigglypuffRL.common.policies import MlpPolicy
from jigglypuffRL.common.values import MlpValue


class MlpActorCritic(BaseActorCritic):
    """
    MLP Actor Critic
    :param s_dim: (int) state dimension of environment
    :param a_dim: (int) action dimension of environment
    :param hidden: (tuple or list) sizes of hidden layers
    :param val_type: (str) type of value function.
        'V' for V(s), 'Qs' for Q(s), 'Qsa' for Q(s,a)
    :param disc: (bool) discrete action space?
    :param det: (bool) deterministic policy?
    """

    def __init__(
        self,
        s_dim,
        a_dim,
        hidden=(32, 32),
        val_type="V",
        disc=True,
        det=True,
        *args,
        **kwargs
    ):
        super(MlpActorCritic, self).__init__(disc, det)

        self.actor = MlpPolicy(s_dim, a_dim, hidden, disc, det, **kwargs)
        self.critic = MlpValue(s_dim, a_dim, val_type, hidden)


ac_registry = {"mlp": MlpActorCritic}


def get_ac_from_name(name_):
    if name_ in ac_registry:
        return ac_registry[name_]
    raise NotImplementedError
