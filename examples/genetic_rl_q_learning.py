import argparse
from typing import Tuple
import gc

import gym
from genrl.classical.qlearning import QLearning
from genrl.classical.common import Trainer

import random
from copy import deepcopy


# Code inspired from https://github.com/harvitronix/neural-network-genetic-algorithm

def get_params_agent(params_selected, agent):
    '''
    To create an agent with the specific parameters

    Args:
        params_selected (Dict): Dictionary of hyperparameters for the agent
        agent: Rl agent

    Returns:
        agent : A new agent with the hyperparameters mention by params_selected


    '''
    new_agent = deepcopy(agent)
    agent_hyperparameters = new_agent.get_hyperparams().keys()

    for key in params_selected:
        if key in agent_hyperparameters:
            exec('new_agent.'+ key + '=' + str(params_selected[key]))
        else:
            raise Exception("Hyperparameter key doesn't match")

    return new_agent

def set_params_agent(agent, parameter, parameter_value):
    '''
    Modify the agent's parameter

    Args:
        agent: Rl agent
        parameter (str): Agent parameter to change
        parameter_value (float):Parameter value of the agent parameter


    '''
    agent_hyperparameters = agent.get_hyperparams().keys()

    if parameter in agent_hyperparameters:
        exec('agent.' + parameter + '=' + str(parameter_value))
    else:
        raise Exception("Hyperparameter key doesn't match")

    return agent

def create_random_agent(parameter_choices, agent):
    '''
    Create new random agent given the parameter choices

    Args:
        parameter_choices(dict) : Dict of all the parameter choices
        agent : RL agent

    Return:
        rnd_agent: New random agent created using parameter_choices
    '''

    rnd_agent = deepcopy(agent)

    agent_hyperparameters = agent.get_hyperparams().keys()

    for key in parameter_choices:
        # value for the parameter mentioned by 'key'
        value = random.choice(parameter_choices[key])

        if key in agent_hyperparameters:
            exec('rnd_agent.'+ key + '=' + str(value))
        else:
            raise Exception("Hyperparameter key doesn't match")

    return rnd_agent

class GeneticOptimizer():

    def __init__(self,
                 parameter_choices,
                 retain = 0.4,
                 random_select = 0.1,
                 mutate_chance = 0.2):

        '''
        Create a Genetic Optimizer for GenRL

        Args:
            parameter_choices(dict) : Possible network parameters
            retain (float) : Percentage of population to retain after each generation
            random_select (float): Probability of a rejected network remaining in the population
            mutate_chance (float): Probability a network will be randomly mutated
        '''

        self.parameter_choices = parameter_choices
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance

    def initialize_population(self, no_of_parents, agent):
        '''
        Create a population of random networks

        Args:
            no_of_parents(int): Number of parents in each generation (size of the population)
            agent (BaseAgent): Generic Agent Object
        Returns:
            (list) : Population of agents
        '''

        population_agents = []

        # looping to create no_of_parents agents
        for _ in range(no_of_parents):

            agent = create_random_agent(self.parameter_choices, agent)

            population_agents.append(agent)

        return population_agents


    def breed(self, mother, father):
        '''
            Make children as part of their parents

            Args:
                mother: agent
                father: agent

            Return:
                List of 2 children (agents)

        '''

        mother_params = mother.get_hyperparams()
        father_params = father.get_hyperparams()

        children = []

        for _ in range(2):

            child_params = {}

            for key in self.parameter_choices:
                child_params[key] = random.choice([mother_params[key], father_params[key]])

            child_agent = get_params_agent(child_params, father)


            children.append(child_agent)

        return children

    def mutate(self, agent):
        '''
        Randomly mutates one part of the agent

        Args:
            agent(BaseAgent): The RL agent
        '''

        # choose a hyperparameter to mutate
        mutation = random.choice(list(self.parameter_choices.keys()))

        # mutate the chosen hyperparameter

        #randomly choose the value
        mutation_value = random.choice(self.parameter_choices[mutation])
        #set the value in the agent
        agent = set_params_agent(agent, mutation, mutation_value)

        return agent

    def fitness(self,agent):
        '''
        Return the mean rewards, which is our fitness function
        '''

        return agent.mean_reward

    def grade(self, population):
        '''
        Average fitness of the population
        '''
        summed = sum([self.fitness(agent) for agent in population])
        return summed / float(len(population))

    def evolve(self, population):
        '''
        Evolve the population of the network

        Args:
            population(list): A list of agents

        '''

        #Get scores for each network
        graded = [(self.fitness(agent), agent) for agent in population]

        #sort of basis of the score
        graded = [x[1] for x in sorted(graded, key=lambda x:x[0], reverse = True)]

        #get the number we want to kep for next gen
        retain_length = int(len(graded) * self.retain)

        #parents we want to retain
        parents = graded[:retain_length]
        to_be_deleted = []

        #for left out individuals we randomly keep some
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
            else:
                to_be_deleted.append(individual)

        #DELETE ALL OTHERS in to_be_deleted
        #done to avoid any memory leak
        for individual in to_be_deleted:
            del individual
        del to_be_deleted
        gc.collect()

        #how many spots left to fill in the population
        parents_length = len(parents)
        children_length = len(population) - len(parents)

        children = []

        while len(children) < children_length:

            if parents_length == 1:
                male = parents[0]
                babies = babies = self.breed(male, male)

                for baby in babies:
                    if len(children) < children_length:
                        children.append(baby)

            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)

                for baby in babies:
                    if len(children) < children_length:
                        children.append(baby)

        parents.extend(children)

        return parents


def train_population(agents,
                     envirnment,
                     args):
    '''
    Train all the agents in the population

    Args:
        agents (List) : List of agent
        envirnment: Gym envirnment

    '''

    # logger = get_logger(args.log)

    for i, agent in enumerate(agents):

        print(f"Training {i}th agent in current generation")

        trainer = Trainer(agent,
                          envirnment,
                          mode="dyna",
                          model="tabular",
                          n_episodes=1000,
                          start_steps=50)

        trainer.train()

        del trainer
        print('-'*80)




def generate(generations,
             no_of_parents,
             agent_parameter_choices,
             envirnment,
             generic_agent,
             args):
    '''
    Genetic Algorithm for RL

    Args:
        generations (int): No of generations
        no_of_parents(int): No of agents in a generation
        agent_parameter_choices(Dict): Parameter choices for the agent
        envirnment: Gym Envirnment
        generic_agent : RL Agent to be tuned


    '''

    optimizer = GeneticOptimizer(agent_parameter_choices)
    agents = optimizer.initialize_population(no_of_parents, generic_agent)

    #evolve the generation
    for i in range(generations):

        print(f"Doing generation {i}/{generations}")

        #Train the agents
        train_population(agents, envirnment, args)

        #get average fitness of the generation
        avg_reward = optimizer.grade(agents)

        print(f"Generation avg reward:{avg_reward}")
        print("-"*50)

        #Evolve the generation
        if i != generations - 1:
            agents = optimizer.evolve(agents)

    #sort our final population
    agents = sorted(agents, key=lambda x: optimizer.fitness(x), reverse=True)

    #print rewards of top 5
    for i in range(5):
        print(f"Top {i+1} agent reward: {optimizer.fitness(agents[i])}")
        print(f"Hyperparameters : {agents[i].get_hyperparams()}")


def main(args):
    env = gym.make("FrozenLake-v0")

    #defining a generic agent that we want to be optimized
    generic_agent = QLearning(env)

    #Possible parameter choices
    agent_parameter_choices = {
        'epsilon': [0.95, 0.9, 0.8, 0.7, 0.6],
        'gamma': [0.98, 0.95, 0.90, 0.80],
        'lr' : [0.005, 0.01, 0.02, 0.04, 0.08, 0.1]
    }

    generate(args.generations,
             args.population,
             agent_parameter_choices,
             env,
             generic_agent,
             args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL algorithms")

    parser.add_argument("--population", help="No. of agents in a generation", default=10, type=int)
    parser.add_argument("--generations", help="No. of generations", default=10, type=int)

    args = parser.parse_args()

    main(args)




