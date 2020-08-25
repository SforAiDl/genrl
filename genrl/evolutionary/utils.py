import random
from copy import deepcopy


def get_params_agent(params_selected, agent):
    """
    To create an agent with the specific parameters

    Args:
        params_selected (Dict): Dictionary of hyperparameters for the agent
        agent: Rl agent

    Returns:
        agent : A new agent with the hyperparameters mention by params_selected


    """
    new_agent = deepcopy(agent)
    agent_hyperparameters = new_agent.get_hyperparams().keys()

    for key in params_selected:
        if key in agent_hyperparameters:
            exec("new_agent." + key + "=" + str(params_selected[key]))
        else:
            raise Exception("Hyperparameter key doesn't match")

    return new_agent


def set_params_agent(agent, parameter, parameter_value):
    """
    Modify the agent's parameter

    Args:
        agent: Rl agent
        parameter (str): Agent parameter to change
        parameter_value (float):Parameter value of the agent parameter


    """
    agent_hyperparameters = agent.get_hyperparams().keys()

    if parameter in agent_hyperparameters:
        exec("agent." + parameter + "=" + str(parameter_value))
    else:
        raise Exception("Hyperparameter key doesn't match")

    return agent


def create_random_agent(parameter_choices, agent):
    """
    Create new random agent given the parameter choices

    Args:
        parameter_choices(dict) : Dict of all the parameter choices
        agent : RL agent

    Return:
        rnd_agent: New random agent created using parameter_choices
    """

    rnd_agent = deepcopy(agent)

    agent_hyperparameters = agent.get_hyperparams().keys()

    for key in parameter_choices:
        # value for the parameter mentioned by 'key'
        value = random.choice(parameter_choices[key])

        if key in agent_hyperparameters:
            exec("rnd_agent." + key + "=" + str(value))
        else:
            raise Exception("Hyperparameter key doesn't match")

    return rnd_agent
