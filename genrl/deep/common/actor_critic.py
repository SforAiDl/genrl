from gym import spaces
from .base import BaseActorCritic
from .policies import MlpPolicy
from .values import MlpValue
from typing import Tuple


class MlpActorCritic(BaseActorCritic):
    """
    MLP Actor Critic

    :param state_dim: State dimensions of the environment
    :param action_dim: Action dimensions of the environment
    :param hidden: Sizes of hidden layers
    :param val_type: Specifies type of value function: \
"V" for V(s), "Qs" for Q(s), "Qsa" for Q(s,a)
    :param discrete: True if action space is discrete, else False
    :type state_dim: int
    :type action_dim: int
    :type hidden: tuple or list
    :type val_type: str
    :type discrete: bool
    """

    def __init__(
        self,
        state_dim: spaces.Space,
        action_dim: spaces.Space,
        hidden: Tuple = (32, 32),
        val_type: str = "V",
        discrete: bool =True,
        *args,
        **kwargs
    ):
        super(MlpActorCritic, self).__init__()

        self.actor = MlpPolicy(state_dim, action_dim, hidden, discrete, **kwargs)
        self.critic = MlpValue(state_dim, action_dim, val_type, hidden)


actor_critic_registry = {"mlp": MlpActorCritic}


def get_actor_critic_from_name(name_: str):
    """
    Returns Actor Critic given the type of the Actor Critic

    :param ac_name: Name of the policy needed
    :type ac_name: str
    :returns: Actor Critic class to be used
    """
    if name_ in actor_critic_registry:
        return actor_critic_registry[name_]
    raise NotImplementedError
