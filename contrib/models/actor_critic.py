import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.autograd as autograd
from typing import Tuple, Type, Union
import numpy as np

import sys
sys.path.append("/home/aditya/Desktop/genrl/genrl")
from contrib.utils.models import SharedMLP, MLP


class BaseActorCritic(nn.Module):
    """
    Basic implementation of a general Actor Critic
    """

    def __init__(self):
        super(BaseActorCritic, self).__init__()

        self.actor = None
        self.critic = None
        self.actorcritic = None

    def get_action(
        self, state: torch.Tensor, one_hot: bool = False, deterministic: bool = False
    ) -> torch.Tensor:
        """
                Get action from the Actor based on input

                :param state: The state being passed as input to the Actor
                :param deterministic: (True if the action space is deterministic,
        else False)
                :type state: Tensor
                :type deterministic: boolean
                :returns: action
        """
        state = torch.as_tensor(state).float()
        if self.actor == None:
            return self.actorcritic.get_action(None,state,deterministic=deterministic)
        return self.actor.get_action(state, deterministic=deterministic)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value from the Critic based on input

        :param state: Input to the Critic
        :type state: Tensor
        :returns: value
        """
        state = torch.as_tensor(state).float()
        if self.critic == None:
            return self.actorcritic.get_value(state,None)
        return self.critic.get_value(state)



class SharedActorCritic(BaseActorCritic):
    def __init__(self, critic_prev,actor_prev,shared,critic_post,actor_post,weight_init,activation_func):
        super(SharedActorCritic, self).__init__()

        self.actorcritic = SharedMLP(critic_prev,actor_prev,shared,critic_post,actor_post,weight_init,activation_func)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state_critic,state_action):

        if state_critic is not None:

            for i in range(len(self.actorcritic.network1_prev)):
                if self.actorcritic.activation is not None:
                    state_critic = self.actorcritic.activation(self.actorcritic.network1_prev[i](state_critic))
                else:
                    state_critic = self.actorcritic.network1_prev[i](state_critic)

            for i in range(len(self.actorcritic.shared)):
                if self.actorcritic.activation is not None:
                    state_critic = self.actorcritic.activation(self.actorcritic.shared[i](state_critic))
                else:
                    state_critic = self.actorcritic.shared[i](state_critic)

            for i in range(len(self.actorcritic.network1_post)):
                if self.actorcritic.activation is not None:
                    state_critic = self.actorcritic.activation(self.actorcritic.network1_post[i](state_critic))
                else:
                    state_critic = self.actorcritic.network1_post[i](state_critic)

            return state_critic

        if state_action is not None:

            for i in range(len(self.actorcritic.network2_prev)):
                if self.actorcritic.activation is not None:
                    state_action = self.actorcritic.activation(self.actorcritic.network2_prev[i](state_action))
                else:
                    state_action = self.actorcritic.network2_prev[i](state_action)

            for i in range(len(self.actorcritic.shared)):
                if self.actorcritic.activation is not None:
                    state_action = self.actorcritic.activation(self.actorcritic.shared[i](state_action))
                else:
                    state_action = self.actorcritic.shared[i](state_action)

            for i in range(len(self.actorcritic.network2_post)):
                if self.actorcritic.activations is not None:
                    state_action = self.actorcritic.activation(self.actorcritic.network2_post[i](state_action))
                else:
                    state_action = self.actorcritic.network2_post[i](state_action)

            return state_action



    def get_action(self, state, one_hot=False, deterministic=False):
        # state = torch.FloatTensor(state).to(self.device)
        logits = self.forward(None,state)
        if one_hot:
            if deterministic:
                logits = self.onehot_from_logits(logits,eps=1.0)
            else:
                logits = self.onehot_from_logits(logits,eps=0.0)
            return logits

        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)
        if deterministic:
            index = torch.argmax(probs)
        else:
            index = probs.sample().cpu().detach().item()
        return index

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ]
        # chooses between best and random actions using epsilon greedy
        return torch.stack(
            [
                argmax_acs[i] if r > eps else rand_acs[i]
                for i, r in enumerate(torch.rand(logits.shape[0]))
            ]
        )

    def get_value(self, state):
        # state = torch.FloatTensor(state).to(self.device)
        value = self.forward(state,None)
        return value


class Actor(BaseActorCritic):
    def __init__(self, layer_sizes,weight_init,activation_func):
        super(Actor, self).__init__()

        self.actor = MLP(layer_sizes,weight_init,activation_func,-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, policy):

        for i in range(len(self.actor.model)):
            if self.actor.activation is not None:
                policy = self.actor.activation(self.actor.model[i](policy))
            else:
                policy = self.actor.model[i](policy)

        return policy



    def get_action(self, state, one_hot=False, deterministic=False):
        # state = torch.FloatTensor(state).to(self.device)
        logits = self.forward(state)
        if one_hot:
            if deterministic:
                logits = self.onehot_from_logits(logits,eps=1.0)
            else:
                logits = self.onehot_from_logits(logits,eps=0.0)
            return logits

        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)
        if deterministic:
            index = torch.argmax(probs)
        else:
            index = probs.sample().cpu().detach().item()
        return index

    def onehot_from_logits(self, logits, eps=0.0):
        # get best (according to current policy) actions in one-hot form
        argmax_acs = (logits == logits.max(0, keepdim=True)[0]).float()
        if eps == 0.0:
            return argmax_acs
        # get random actions in one-hot form
        rand_acs = torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ]
        # chooses between best and random actions using epsilon greedy
        return torch.stack(
            [
                argmax_acs[i] if r > eps else rand_acs[i]
                for i, r in enumerate(torch.rand(logits.shape[0]))
            ]
        )


class Critic(BaseActorCritic):
    def __init__(self, layer_sizes,weight_init,activation_func,concat_ind):
        super(Critic, self).__init__()

        self.critic = MLP(layer_sizes,weight_init,activation_func,concat_ind)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, value, action, concatenate_index):

        for i in range(len(self.critic.model)):
            if i==concatenate_index:
                value = torch.cat([value,action], 1)
            if self.critic.activation is not None:
                value = self.critic.activation(self.critic.model[i](value))
            else:
                value = self.actor.model[i](value)

        return value



    def get_value(self, state):
        # state = torch.FloatTensor(state).to(self.device)
        value = self.forward(state)
        return value




# model = SharedActorCritic([1,1],[1,1],[1,1],[1,1],[1,1],"xavier_uniform","relu")
# forward = model.forward(torch.Tensor([1]),None)
# print(forward)