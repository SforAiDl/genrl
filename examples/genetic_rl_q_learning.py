import argparse

import gym

from genrl.classical.common import Trainer
from genrl.classical.qlearning import QLearning
from genrl.evolutionary import GeneticHyperparamTuner
from genrl.evolutionary.genetic_hyperparam import generate

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

    optimizer = GATuner(agent_parameter_choices)

    generate(
        optimizer,
        train_population,
        args.generations,
        args.population,
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
