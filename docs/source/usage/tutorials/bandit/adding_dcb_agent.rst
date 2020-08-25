Adding a new Deep Contextual Bandit Agent
=========================================

The ``bandit`` submodule like all of ``genrl`` has been designed to be
easily extensible for custom additions. This tutorial will show how to
create a deep contextual bandit agent which will work with the rest of
``genrl.bandit``

For the purpose of this tutorial we will consider a simple neural
network based agent. Although this is a simplictic agent, implementation
of any level of agent will need to have the following steps.

To start off with lets import necessary modules and make a class which
inherits from ``genrl.agents.bandits.contextual.base.DCBAgent``

.. code:: python

    from typing import Optional

    import torch

    from genrl.agents.bandits.contextual.base import DCBAgent
    from genrl.agents.bandits.contextual.common import NeuralBanditModel, TransitionDB
    from genrl.utils.data_bandits.base import DataBasedBandit

    class NeuralAgent(DCBAgent):
        """Deep contextual bandit agent based on a neural network."""

        def __init__(self, bandit: DataBasedBandit, **kwargs):

        def select_action(self, context: torch.Tensor) -> int:

        def update_db(self, context: torch.Tensor, action: int, reward: int):

        def update_params(
            self,
            action: Optional[int] = None,
            batch_size: int = 512,
            train_epochs: int = 20,
        ):

We will need to implement ``__init__``, ``select_action``, ``update_db``
and ``update_param`` to make the class functional.

Lets start off with ``__init__``. Here we will need to initialise some
required parameters (``init_pulls``, ``eval_with_dropout``, ``t`` and
``update_count``) along with our transition database and the neural
network. For the neural network, you can use the ``NeuralBanditModel``
class. It packages together many of the functionalities a neural network
might require. Refer to the docs for more details.

.. code:: python

        def __init__(self, bandit: DataBasedBandit, **kwargs):
            super(NeuralAgent, self).__init__(bandit, **kwargs)
            self.model = (
                NeuralBanditModel(
                    context_dim=self.context_dim,
                    n_actions=self.n_actions,
                    **kwargs
                )
                .to(torch.float)
                .to(self.device)
            )
            self.eval_with_dropout = kwargs.get("eval_with_dropout", False)
            self.db = TransitionDB(self.device)
            self.t = 0
            self.update_count = 0

For the select action function, the agent will pass the context vector
through the neural network to produce logits for each action. It will
then select the action with highest logit value. Note that it must also
increment the timestep, and if take every action atleast ``init_pulls``
number of times initially.

.. code:: python

        def select_action(self, context: torch.Tensor) -> int:
            """Selects action for a given context"""
            self.model.use_dropout = self.eval_with_dropout
            self.t += 1
            if self.t < self.n_actions * self.init_pulls:
                return torch.tensor(
                    self.t % self.n_actions, device=self.device, dtype=torch.int
                )

            results = self.model(context)
            action = torch.argmax(results["pred_rewards"]).to(torch.int)
            return action

For updating the databse we can use the ``add`` method of
``TransitionDB`` class.

.. code:: python

        def update_db(self, context: torch.Tensor, action: int, reward: int):
            """Updates transition database."""
            self.db.add(context, action, reward)

In ``update_params`` we need to train the model on the observations seen
so far. Since the ``NeuralBanditModel`` class already hass a train
function, we just need to call that. However if you are writing your own
model, this is where the updates to the parameters would happen.

.. code:: python

        def update_params(
            self,
            action: Optional[int] = None,
            batch_size: int = 512,
            train_epochs: int = 20,
        ):
            """Update parameters of the agent."""
            self.update_count += 1
            self.model.train_model(self.db, train_epochs, batch_size)

Note that some of these functions have unused arguments. The signatures
have been decided so as such to ensure generality over all classes of
algorithms.

Once you are done with the above, you can use the ``NeuralAgent`` class
like you would any other agent from ``genrl.bandit``. You can use it
with any of the bandits as well as training it with
`genrl.bandit.DCBTrainer <../../../api/common/bandit.html#module-genrl.bandit.trainer>`__.
