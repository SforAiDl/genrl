=================
SARSA using GenRL
=================

What is SARSA? 
==============

SARSA is an acronym for State-Action-Reward-State-Action. It is an on-policy TD control method. Our aim is basically to estimate the Q-value or the utility value for state-action pair using the TD update rule given below. 

.. math::

    Q(S_{t}, A_{t}) = Q(S_{t}, A_{t}) + \alpha * [R_{t+1} + \gamma * Q(S_{t+1}, A_{t+2}) - Q(S_{t}, A_{t})]

FrozenLake-v0 environment 
=========================

So to demonstrate how easy it is to train a SARSA approach in GenRL, we are taking a very simple gym environment. 

Description of the environment (from the documentation) - 

"The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following:

.. code-block:: text

    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)


The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise."

Code 
====

Let's import all the usefull stuff first. 

.. code-block:: python

    import gym
    from genrl import SARSA                                 # for the agent 
    from genrl.classical.common import Trainer              # for training the agent 

Now that we have imported all the necessary stuff let's go ahead and define the environment, the agent and an object for the Trainer class. 

.. code-block:: python

    env = gym.make("FrozenLake-v0")                               
    agent = SARSA(env, gamma=0.6, lr=0.1, epsilon=0.1)
    trainer = Trainer(
        agent,
        env,
        model="tabular",
        n_episodes=3000,
        start_steps=100,
        evaluate_frequency=100,
    )  

Great so far so good! Now moving towards the training process it is just calling the train method in the trainer class. 

.. code-block:: python

    trainer.train()
    trainer.evaluate()

That's it! You have successfully trained a SARSA agent. You can now go ahead and play with your own environments using GenRL!