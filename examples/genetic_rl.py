import argparse

from genrl import A2C, PPO1
from genrl.deep.common import OnPolicyTrainer
from genrl.deep.common.actor_critic import MlpActorCritic
from genrl.deep.common.utils import get_env_properties
from genrl.environments import VectorEnv
from genrl.evolutionary import GeneticHyperparamTuner

# """
# Okay so parameters to tune:-
#  - layers
#  - lr_policy
#  - lr_value
#  - clip param
#  - entropy coeff
#  - value coeff
#  - gamma
# """


def get_logger(log):
    if "," not in log:
        return [log]
    else:
        log = log.split(",")
        if "" in log or " " in log:
            log = [i for i in log if i != ""]
            log = [i for i in log if i != " "]
        return log


# Code inspired from https://github.com/harvitronix/neural-network-genetic-algorithm


class GATuner(GeneticHyperparamTuner):
    def fitness(self, agent):
        """
        Return the mean rewards, which is our fitness function
        """

        return agent.get_logging_params()["mean_reward"]


def train_population(agents, envirnment, args):
    """
    Train all the agents in the population

    Args:
        agents (List) : List of agent
        envirnment: Gym envirnment

    """

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


def main(args):
    env = VectorEnv(
        args.env, n_envs=args.n_envs, parallel=not args.serial, env_type=args.env_type
    )

    input_dim, action_dim, discrete, action_lim = get_env_properties(env, "mlp")

    network = MlpActorCritic(
        input_dim,
        action_dim,
        (1, 1),  # layers
        (1, 1),
        "V",  # type of value function
        discrete,
        action_lim=action_lim,
        activation="relu",
    )

    generic_agent = A2C(network, env, rollout_size=args.rollout_size)

    agent_parameter_choices = {
        "gamma": [12, 121],
        # 'clip_param': [0.2, 0.3],
        # 'lr_policy': [0.001, 0.002],
        # 'lr_value': [0.001, 0.002]
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
    parser = argparse.ArgumentParser(description="Train Deep RL algorithms")
    # parser.add_argument("-a", "--algo", help="Which Algo to train", default="ppo", type=str)
    parser.add_argument(
        "-e", "--env", help="Which env to train on", default="CartPole-v0", type=str
    )
    parser.add_argument(
        "--env-type", help="What kind of env is it", default="gym", type=str
    )
    parser.add_argument(
        "-n",
        "--n-envs",
        help="Number of vectorized envs to train on",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--serial",
        help="Vectorized envs should be serial or parallel",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--epochs", help="How many epochs to train on", default=20, type=int
    )
    parser.add_argument(
        "--render",
        help="Should the env be rendered",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--log", help="Comma separated string of logs", default="stdout", type=str
    )
    parser.add_argument(
        "--arch", help="Which architecture mlp/cnn for now", default="mlp", type=str
    )
    parser.add_argument("--log-interval", help="Set Log interval", default=50, type=int)
    parser.add_argument("--batch-size", help="Batch Size", default=128, type=int)
    parser.add_argument(
        "--population", help="No. of agents in a generation", default=10, type=int
    )
    parser.add_argument("--generations", help="No. of generations", default=5, type=int)

    offpolicyargs = parser.add_argument_group("Off Policy Args")
    offpolicyargs.add_argument(
        "-ws", "--warmup-steps", help="Warmup steps", default=10000, type=int
    )
    offpolicyargs.add_argument(
        "--replay-size", help="Replay Buffer Size", default=1000, type=int
    )

    onpolicyargs = parser.add_argument_group("On Policy Args")
    onpolicyargs.add_argument(
        "--rollout-size", help="Rollout Buffer Size", default=2048, type=int
    )

    args = parser.parse_args()

    main(args)
