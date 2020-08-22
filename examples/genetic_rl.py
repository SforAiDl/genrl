import argparse
from typing import Tuple
import gc

from genrl import PPO1
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv
from gym import spaces

from genrl.deep.common.actor_critic import MlpActorCritic
from genrl.deep.common import get_env_properties

import random
from copy import deepcopy
'''
Okay so parameters to tune:- 
 - layers
 - lr_policy 
 - lr_value
 - clip param 
 - entropy coeff
 - value coeff
 - gamma
 - 
'''

def get_logger(log):
    if "," not in log:
        return [log]
    else:
        log = log.split(",")
        if "" in log or " " in log:
            log = [i for i in log if i != ""]
            log = [i for i in log if i != " "]
        return log


# https://github.com/harvitronix/neural-network-genetic-algorithm

def get_params_agent(params_selected, agent):
    '''
    To create an agent with the specific parameters

    '''
    new_agent = deepcopy(agent)
    agent_hyperparameters = new_agent.get_hyperparameters().keys()

    for key in params_selected:
        if key in agent_hyperparameters:
            exec('new_agent.'+ key + '=' + str(params_selected[key]))
        else:
            raise Exception("Hyperparameter key doesn't match")

    return new_agent

def set_params_agent(agent, parameter, parameter_value):
    '''
    Set the agent.parameter to parameter_value

    '''
    agent_hyperparameters = agent.get_hyperparameters().keys()

    if parameter in agent_hyperparameters:
        exec('agent.' + parameter + '=' + str(parameter_value))
    else:
        raise Exception("Hyperparameter key doesn't match")

    return agent

def create_random_agent(parameter_choices, agent):

    rnd_agent = deepcopy(agent)

    agent_hyperparameters = agent.get_hyperparameters().keys()

    for key in parameter_choices:
        # value for the parameter mentioned by 'key'
        value = random.choice(parameter_choices[key])
        # # NOTE PUT HEAVY TRY AND CATCH SATEMENTS
        # '''
        # Cases
        #     - Agents own parameters
        #     - Networks Parameter (Idea dict of dict for networks and
        #     seperate if statement)
        #
        # '''

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

        mother_params = mother.get_hyperparameters()
        father_params = father.get_hyperparameters()

        children = []

        for _ in range(2):

            child_params = {}

            for key in self.nn_param_choices:
                child_params[key] = random.choice([mother_params[key], father_params[key]])

            child_agent = get_params_agent(child_params, father)


            children.append(child_agent)

        return children

    def mutate(self, agent):
        '''
        Randomly mutates one part of the agent

        Args:
            agent(BaseAgent)
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

        return agent.get_logging_params()['mean_reward']

    def grade(self, population):
        '''
        Average fitness of the population
        '''
        summed = ([self.fitness(agent) for agent in population]).sum()
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

    '''

    logger = get_logger(args.log)

    for agent in agents:

        trainer = OnPolicyTrainer(
            agent,
            envirnment,
            logger,
            epochs=args.epochs,
            render=args.render,
            log_interval=args.log_interval,
        )

        trainer.train()




def generate(generations,
             no_of_parents,
             agent_parameter_choices,
             envirnment,
             generic_agent,
             args):
    '''

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


def main(args):
    env = VectorEnv(args.env, n_envs=args.n_envs, parallel=not args.serial, env_type=args.env_type)

    input_dim, action_dim, discrete, action_lim = get_env_properties(env, "mlp")

    network = MlpActorCritic(
        input_dim,
        action_dim,
        (1, 1),  # layers
        "V",  # type of value function
        discrete,
        action_lim=action_lim,
        activation='relu'
    )

    generic_agent = PPO1(
        network,
        env,
        rollout_size=args.rollout_size)

    agent_parameter_choices = {
        'gamma': [12, 121],
        'clip_param': [0.2, 0.3],
        # 'lr_policy': [0.001, 0.002],
        # 'lr_value': [0.001, 0.002]
    }

    generate(args.generations,
             args.population,
             agent_parameter_choices,
             env,
             generic_agent,
             args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Deep RL algorithms")
    # parser.add_argument("-a", "--algo", help="Which Algo to train", default="ppo", type=str)
    parser.add_argument("-e", "--env", help="Which env to train on", default="CartPole-v0", type=str)
    parser.add_argument("--env-type", help="What kind of env is it", default="gym", type=str)
    parser.add_argument("-n", "--n-envs", help="Number of vectorized envs to train on", default=2, type=int)
    parser.add_argument("--serial", help="Vectorized envs should be serial or parallel", default=True, type=bool)
    parser.add_argument("--epochs", help="How many epochs to train on", default=20, type=int)
    parser.add_argument("--render", help="Should the env be rendered", default=False, action="store_true")
    parser.add_argument("--log", help="Comma separated string of logs", default="stdout", type=str)
    parser.add_argument("--arch", help="Which architecture mlp/cnn for now", default="mlp", type=str)
    parser.add_argument("--log-interval", help="Set Log interval", default=50, type=int)
    parser.add_argument("--batch-size", help="Batch Size", default=128, type=int)
    parser.add_argument("--population", help="No. of agents in a generation", default=10, type=int)
    parser.add_argument("--generations", help="No. of generations", default=5, type=int)

    offpolicyargs = parser.add_argument_group("Off Policy Args")
    offpolicyargs.add_argument("-ws", "--warmup-steps", help="Warmup steps", default=10000, type=int)
    offpolicyargs.add_argument("--replay-size", help="Replay Buffer Size", default=1000, type=int)

    onpolicyargs = parser.add_argument_group("On Policy Args")
    onpolicyargs.add_argument("--rollout-size", help="Rollout Buffer Size", default=2048, type=int)

    args = parser.parse_args()

    main(args)




