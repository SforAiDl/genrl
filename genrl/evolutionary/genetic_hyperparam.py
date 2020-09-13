import gc
import random
from typing import Dict

from genrl.evolutionary.utils import (
    create_random_agent,
    get_params_agent,
    set_params_agent,
)

"""
Code is heavily inspired from https://github.com/harvitronix/neural-network-genetic-algorithm
"""


class GeneticHyperparamTuner:
    def __init__(
        self,
        parameter_choices: [Dict],
        retain: float = 0.4,
        random_select: float = 0.1,
        mutate_chance: float = 0.2,
    ):

        """
        Create a Genetic Hyperparameter Tuner for GenRL

        Args:
            parameter_choices(dict) : Possible network parameters
            retain (float) : Percentage of population to retain after each generation
            random_select (float): Probability of a rejected network remaining in the population
            mutate_chance (float): Probability a network will be randomly mutated
        """

        self.parameter_choices = parameter_choices
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance

    def initialize_population(self, no_of_parents, agent):
        """
        Create a population of random networks

        Args:
            no_of_parents(int): Number of parents in each generation (size of the population)
            agent (BaseAgent): Generic Agent Object
        Returns:
            (list) : Population of agents
        """

        population_agents = []

        # looping to create no_of_parents agents
        for _ in range(no_of_parents):

            agent = create_random_agent(self.parameter_choices, agent)

            population_agents.append(agent)

        return population_agents

    def breed(self, mother, father):
        """
        Make children as part of their parents

        Args:
            mother: agent
            father: agent

        Return:
            List of 2 children (agents)

        """

        mother_params = mother.get_hyperparams()
        father_params = father.get_hyperparams()

        children = []

        for _ in range(2):

            child_params = {}

            for key in self.parameter_choices:
                child_params[key] = random.choice(
                    [mother_params[key], father_params[key]]
                )

            child_agent = get_params_agent(child_params, father)

            children.append(child_agent)

        return children

    def mutate(self, agent):
        """
        Randomly mutates one part of the agent

        Args:
            agent(BaseAgent): The RL agent
        """

        # choose a hyperparameter to mutate
        mutation = random.choice(list(self.parameter_choices.keys()))

        # mutate the chosen hyperparameter

        # randomly choose the value
        mutation_value = random.choice(self.parameter_choices[mutation])
        # set the value in the agent
        agent = set_params_agent(agent, mutation, mutation_value)

        return agent

    def fitness(self, agent):
        """
        Return the mean rewards, which is our fitness function
        """

        return NotImplementedError

    def grade(self, population):
        """
        Average fitness of the population
        """
        summed = sum([self.fitness(agent) for agent in population])
        return summed / float(len(population))

    def evolve(self, population):
        """
        Evolve the population of the network

        Args:
            population(list): A list of agents

        """

        # Get scores for each network
        graded = [(self.fitness(agent), agent) for agent in population]

        # sort of basis of the score
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # get the number we want to kep for next gen
        retain_length = int(len(graded) * self.retain)

        # parents we want to retain
        parents = graded[:retain_length]
        to_be_deleted = []

        # for left out individuals we randomly keep some
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
            else:
                to_be_deleted.append(individual)

        # DELETE ALL OTHERS in to_be_deleted
        # done to avoid any memory leak
        del to_be_deleted[:]
        del to_be_deleted
        gc.collect()

        # how many spots left to fill in the population
        parents_length = len(parents)
        children_length = len(population) - len(parents)

        children = []

        while len(children) < children_length:

            if parents_length == 1:
                male = parents[0]
                babies = self.breed(male, male)

                for baby in babies:
                    if len(children) < children_length:
                        children.append(baby)

            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)

                for baby in babies:
                    if len(children) < children_length:
                        children.append(baby)

        parents.extend(children)

        return parents
