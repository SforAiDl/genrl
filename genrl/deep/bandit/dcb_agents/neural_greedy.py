from typing import List, Optional

import torch

from ..data_bandits import DataBasedBandit
from .common import NeuralBanditModel, TransitionDB
from .dcb_agent import DCBAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class NeuralGreedyAgent(DCBAgent):
    """Deep contextual bandit agent using epsilon greedy with a neural network.

    Args:
        bandit (DataBasedBandit): The bandit to solve
        init_pulls (int, optional): Number of times to select each action initially.
            Defaults to 3.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
            Defaults to [50, 50].
        init_lr (float, optional): Initial learning rate. Defaults to 0.1.
        lr_decay (float, optional): Decay rate for learning rate. Defaults to 0.5.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to True.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
            Defaults to 0.5.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        eval_with_dropout (bool, optional): Whether or not to use dropout at inference.
            Defaults to False.
        epsilon (float, optional): Probability of selecting a random action. Defaults to 0.0.
    """

    def __init__(
        self,
        bandit: DataBasedBandit,
        init_pulls: int = 3,
        hidden_dims: List[int] = [50, 50],
        init_lr: float = 0.1,
        lr_decay: float = 0.5,
        lr_reset: bool = True,
        max_grad_norm: float = 0.5,
        dropout_p: Optional[float] = None,
        eval_with_dropout: bool = False,
        epsilon: float = 0.0,
    ):
        super(NeuralGreedyAgent, self).__init__(bandit)
        self.init_pulls = init_pulls
        self.model = NeuralBanditModel(
            self.context_dim,
            hidden_dims,
            self.n_actions,
            init_lr,
            max_grad_norm,
            lr_decay,
            lr_reset,
            dropout_p,
        )
        self.eval_with_dropout = eval_with_dropout
        self.epsilon = epsilon
        self.db = TransitionDB()
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by computing a forward pass through network
        with an epsillon probability of selecting a random action.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(self.t % self.n_actions, device=device, dtype=torch.int)
        if torch.randn(1) < self.epsilon:
            action = torch.randint(self.n_actions, size=(1,))
        else:
            _, predicted_rewards = self.model(context)
            action = torch.argmax(predicted_rewards).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db.add(context, action, reward)

    def update_params(
        self,
        batch_size: int = 512,
        train_epochs: int = 20,
        action: Optional[int] = None,
    ):
        """Update parameters of the agent.

        Trains neural network.

        Args:
            batch_size (int): Size of batch to update parameters with.
            train_epochs (int): Epochs to train neural network for.
            action (Optional[int], optional): Action to update the parameters for.
                Defaults to None.
        """
        self.update_count += 1
        self.model.train_model(self.db, train_epochs, batch_size)
