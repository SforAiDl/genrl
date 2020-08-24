==============================
Tuning using Genetic Algorithm
==============================


GenRL provides various algorithms to optimize agents. Each algorithm (Eg. QLearning, PPO1, etc) have their own unique set of hyperparameters. Often the simplest method to boost an agents performance is to tune these hyperparameters.

Although a common methodology in machine learning is to do **Grid Search** among the possible values of the hyperparameters. It searches all th possible combination in the search space. But is often very time consuming [1]. **Genetic Algorithms** a good alternative to find the optimum hyperparameters in fraction of the time.

This tutorial will guide the reader to use Genetic Algorithms to tune hyperparamets in an agent for the GenRL library.


Background
==========

GA is a search heuristic that is inspired from the process of evolution. It uses evolutionary Strategies (like Mutation, Crossover and Selection) to evolve the population and find the best candidates. [3]

GAs encode the decision variables of a search problem into a **chromosome**. Each parameter in the choromosome is referred to as a **gene** and value of each gene is referred to as an **allele**. To evolve the agents through natural selection, a fitness function is needed. The fitness function measures the agents relative fitness to distinguish a good agent from a bad one in the population.

In this tutorial we will evolve a QLearning agent in the Frozen Lake envirnment available in gym. Our chromosome will look something as follows:-

To evolve the population of our agents we do the following [4]:-

#. **Initialization** : Random Initialization of the N agents accross the search space

#. **Evaluation** : The fitness of all the Agents is evaluated.
        - Here we would first need to train each agent in the population and then evaluate our fitness function. Here we use the Mean Reward in a episode as the fitness function.

#. **Selection** : Allocate more copies of the solution with higher fitness value.
        - Sort the agents in the current population according to their fitness (Mean Rward). Keep some percent of the top agents for the next generation and for breeding children.
        - One may keep some percentage of randomly selected bad agents to avoid getting stuck at local maxima.
        - Random two networks are choosen as parents for the next step.

#. **Recombination** : Combine parts of two parent solutions to create an offspring ( which will posibly be a better solution).
        - Random combinations of hyperparameters is done from randomly selected parents to ceate an offspring.

#. **Mutation** : Mutation locally modifies the solution by randomly alterring the agents trait.
            - Randomly change a single parameter in some of the agents.


#. **Replacement** : The offspring population replaces the parental population
            - Populate the remaining spots in the population ( Those not filled by the Parents) with these offsprings.


Repeat the steps 2 - 6 to evolve the agent.

Optimizing through API
======================

GenRL an easy way to evolve your agent. You just need to do the following steps:-


Define a class where the fitness function for the agent is defined. Use `GeneticHyperparamTuner` as a base class.

.. code-block:: python

    from genrl.evolutionary import GeneticHyperparamTuner

    class GATuner(GeneticHyperparamTuner):
        def fitness(self, agent):
            """
            Define fitness function for the agent
            """
            return agent.mean_reward

Define a training function to train all the agents in a population. Define the `Trainer` object using you choice of training parameters.

.. code-block:: python

    from genrl.classical.common import Trainer

    def train_population(agents, envirnment, args):
    """
    Train all the agents in the population

    Args:
        agents (List) : List of agent
        envirnment: Gym envirnment

    """

    for i, agent in enumerate(agents):

        print(f"Training {i}th agent in current generation")

        trainer = Trainer(
            agent,
            envirnment,
            mode="dyna",
            model="tabular",
            n_episodes=1000,
            start_steps=50,
        )

        trainer.train()

        del trainer
        print("-" * 80)

Define your environment and a generic agent you want to find the optimum parameters of.

.. code-block:: python

    import gym

    from genrl.classical.qlearning import QLearning

    env = gym.make("FrozenLake-v0")
    # defining a generic agent that we want to be optimized
    generic_agent = QLearning(env)

Define your search space for the parameters you are interested in tuning.

.. code-block:: python

    agent_parameter_choices = {
        "epsilon": [0.95, 0.9, 0.8, 0.7, 0.6],
        "gamma": [0.98, 0.95, 0.90, 0.80],
        "lr": [0.005, 0.01, 0.02, 0.04, 0.08, 0.1],
    }

Define optimizer object and call the `generate` function

.. code-block:: python

    from genrl.evolutionary.genetic_hyperparam import generate

    optimizer = GATuner(agent_parameter_choices)

    generate(
        optimizer,
        train_population,
        args.generations,
        args.population,
        env,
        generic_agent,
        args
    )

Here is the entire code for tuning QLearning agent using GA


.. literalinclude:: ../../../../../examples/genetic_rl_q_learning.py
   :lines: 0-135
   :lineno-start: 0



[1] https://github.com/harvitronix/neural-network-genetic-algorithm

[2] Hyper Parameter Optimization using Genetic
Algorithm on Machine Learning Methods for Online
News Popularity Prediction
Ananto Setyo Wicaksono 1 , Ahmad Afif Supianto 2
Department of Informatics
Faculty of Computer Science, Brawijaya University
Malang, Indonesia

[3] Optimizing Artificial Neural Network Hyperparameters and
Architecture
Ivar Thokle Hovden, ivarth@student.matnat.uio.no, University of Oslo

[4] Sastry K., Goldberg D., Kendall G. (2005) Genetic Algorithms. In: Burke E.K., Kendall G. (eds) Search Methodologies. Springer, Boston, MA. https://doi.org/10.1007/0-387-28356-0_4
