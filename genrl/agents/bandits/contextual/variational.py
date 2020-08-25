import torch

from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.agents.bandits.contextual.common import BayesianNNBanditModel, TransitionDB
from genrl.utils.data_bandits.base import DataBasedBandit


class VariationalAgent(DCBAgent):
    """Deep contextual bandit agent using variation inference.

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
        noise_std (float, optional): Standard deviation of noise in bayesian neural network.
            Defaults to 0.1.
        device (str): Device to use for tensor operations.
            "cpu" for cpu or "cuda" for cuda. Defaults to "cpu".
    """

    def __init__(self, bandit: DataBasedBandit, **kwargs):
        super(VariationalAgent, self).__init__(bandit, kwargs.get("device", "cpu"))
        self.init_pulls = kwargs.get("init_pulls", 3)
        self.model = (
            BayesianNNBanditModel(
                context_dim=self.context_dim,
                hidden_dims=kwargs.get("hidden_dims", [50, 50]),
                n_actions=self.n_actions,
                init_lr=kwargs.get("init_lr", 0.1),
                max_grad_norm=kwargs.get("max_grad_norm", 0.5),
                lr_decay=kwargs.get("lr_decay", 0.5),
                lr_reset=kwargs.get("lr_reset", True),
                dropout_p=kwargs.get("dropout_p", None),
                noise_std=kwargs.get("noise_std", 0.1),
            )
            .to(torch.float)
            .to(self.device)
        )
        self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
        self.db = TransitionDB(self.device)
        self.t = 0
        self.update_count = 0

    def select_action(self, context: torch.Tensor) -> int:
        """Select an action based on given context.

        Selects an action by computing a forward pass through
        the bayesian neural network.

        Args:
            context (torch.Tensor): The context vector to select action for.

        Returns:
            int: The action to take.
        """
        self.model.use_dropout = self.eval_with_dropout
        self.t += 1
        if self.t < self.n_actions * self.init_pulls:
            return torch.tensor(
                self.t % self.n_actions, device=self.device, dtype=torch.int
            )
        results = self.model(context)
        action = torch.argmax(results["pred_rewards"]).to(torch.int)
        return action

    def update_db(self, context: torch.Tensor, action: int, reward: int):
        """Updates transition database with given transition

        Args:
            context (torch.Tensor): Context recieved
            action (int): Action taken
            reward (int): Reward recieved
        """
        self.db.add(context, action, reward)

    def update_params(self, action: int, batch_size: int = 512, train_epochs: int = 20):
        """Update parameters of the agent.

        Trains each neural network in the ensemble.

        Args:
            action (Optional[int], optional): Action to update the parameters for.
                Not applicable in this agent. Defaults to None.
            batch_size (int, optional): Size of batch to update parameters with.
                Defaults to 512
            train_epochs (int, optional): Epochs to train neural network for.
                Defaults to 20
        """
        self.update_count += 1
        self.model.train_model(self.db, train_epochs, batch_size)
