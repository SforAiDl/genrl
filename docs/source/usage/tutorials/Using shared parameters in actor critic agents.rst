Using Shared Parameters in Actor Critic Agents in GenRL
=======================================================

The Actor Critic Agents use two networks, an Actor network to select an action to be taken in the current state, and a
critic network, to estimate the value of the state the agent is currently in. There are two common ways to implement
this actor critic architecture.

The first method - Indpendent Actor and critic networks -

.. code-block:: none

                state
               /     \
    <actor network>   <critic network>
            /            \
        action         value

And the second method - Using a set of shared parameters to extract a feature vector from the state. The actor and the
critic network act on this feature vector to select an action and estimate the value

.. code-block:: none

                state
                  |
              <decoder>
            /           \
   <actor network>     <critic network>
         /                 \
     action               value

GenRL provides support to incorporte this decoder network in all of the actor critic agents through a ``shared_layers``
parameter. ``shared_layers`` takes the sizes of the mlp layers to be used, and ``None`` if no decoder network is to be
used

As an example - in A2C -

.. code-block:: python

    # The imports
    from genrl.agents import A2C
    from genrl.environments import VectorEnv
    from genrl.trainers import OnPolicyTrainer

    # Initializing the environment
    env = VectorEnv("CartPole-v0", 1)

    # Initializing the agent to be used
    algo = A2C(
            "mlp",
            env,
            policy_layers=(128,),
            value_layers=(128,),
            shared_layers=(32, 64),
            rollout_size=128,
        )

    # Finally initializing the trainer and trainer
    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()

The above example uses and mlp of layer sizes (32, 64) as the decoder, and can be visualised as follows -

.. code-block:: none

                state
                  |
                 <32>
                  |
                 <64>
                /    \
             <128>   <128>
              /        \
           action     value