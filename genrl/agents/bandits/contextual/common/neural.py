from typing import Dict

import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F

from genrl.agents.bandits.contextual.common.base_model import Model
from genrl.agents.bandits.contextual.common.transition import TransitionDB


class NeuralBanditModel(Model):
    """Neural Network used in Deep Contextual Bandit Models.

    Args:
        context_dim (int): Length of context vector.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
        n_actions (int): Number of actions that can be selected. Taken as length
            of output vector for network to predict.
        init_lr (float, optional): Initial learning rate.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
        lr_decay (float, optional): Decay rate for learning rate.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to False.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.

    Attributes:
        use_dropout (bool): Indicated whether or not dropout should be used in forward pass.
    """

    def __init__(self, **kwargs):
        super(NeuralBanditModel, self).__init__(nn.Linear, **kwargs)

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes forward pass through the network.

        Args:
            context (torch.Tensor): The context vector to perform forward pass on.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of outputs
        """
        x = context
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if self.dropout_p is not None and self.use_dropout is True:
                x = F.dropout(x, self.dropout_p)

        pred_rewards = self.layers[-1](x)
        return dict(x=x, pred_rewards=pred_rewards)

    def _compute_loss(
        self,
        db: TransitionDB,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        reward_vec: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Computes loss for the model

        Args:
            db (TransitionDB): The database of transitions to train on.
            x (torch.Tensor): Context.
            action_mask (torch.Tensor): Mask of actions taken.
            reward_vec (torch.Tensor): Reward vector recieved.
            batch_size (int): The size of each batch to perform gradient descent on.

        Returns:
            torch.Tensor: The computed loss.
        """
        results = self.forward(x)
        loss = (
            torch.sum(action_mask * (reward_vec - results["pred_rewards"]) ** 2)
            / batch_size
        )
        return loss
