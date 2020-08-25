import argparse

import gym

from genrl.classical.common import Trainer
from genrl.classical.qlearning import QLearning
from genrl.evolutionary import GeneticHyperparamTuner

# Code inspired from https://github.com/harvitronix/neural-network-genetic-algorithm


class GATuner(GeneticHyperparamTuner):
    def fitness(self, agent):
        """
        Define fitness function for the agent
        """
        return agent.mean_reward


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


def generate(
    generations, no_of_parents, agent_parameter_choices, envirnment, generic_agent, args
):
    """
    Genetic Algorithm for RL

    Args:
        generations (int): No of generations
        no_of_parents(int): No of agents in a generation
        agent_parameter_choices(Dict): Parameter choices for the agent
        envirnment: Gym Envirnment
        generic_agent : RL Agent to be tuned


    """

    optimizer = GATuner(agent_parameter_choices)
    agents = optimizer.initialize_population(no_of_parents, generic_agent)

    # evolve the generation
    for i in range(generations):

        print(f"Doing generation {i}/{generations}")

        # Train the agents
        train_population(agents, envirnment, args)

        # get average fitness of the generation
        avg_reward = optimizer.grade(agents)

        print(f"Generation avg reward:{avg_reward}")
        print("-" * 50)

        # Evolve the generation
        if i != generations - 1:
            agents = optimizer.evolve(agents)

    # sort our final population
    agents = sorted(agents, key=lambda x: optimizer.fitness(x), reverse=True)

    # print rewards of top 5
    for i in range(5):
        print(f"Top {i+1} agent reward: {optimizer.fitness(agents[i])}")
        print(f"Hyperparameters : {agents[i].get_hyperparams()}")


def main(args):
    env = gym.make("FrozenLake-v0")

    # defining a generic agent that we want to be optimized
    generic_agent = QLearning(env)

    # Possible parameter choices
    agent_parameter_choices = {
        "epsilon": [0.95, 0.9, 0.8, 0.7, 0.6],
        "gamma": [0.98, 0.95, 0.90, 0.80],
        "lr": [0.005, 0.01, 0.02, 0.04, 0.08, 0.1],
    }

    generate(
        args.generations,
        args.population,
        agent_parameter_choices,
        env,
        generic_agent,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL algorithms")

    parser.add_argument(
        "--population", help="No. of agents in a generation", default=10, type=int
    )
    parser.add_argument(
        "--generations", help="No. of generations", default=10, type=int
    )

    args = parser.parse_args()

    main(args)
