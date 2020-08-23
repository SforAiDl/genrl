from genrl import SAC
from genrl.deep.common import OnPolicyTrainer
from genrl.environments import VectorEnv
from genrl.environments.multiagent import scenarios
from genrl.environments.multiagent.environment import MultiAgentEnv

# env = VectorEnv("Pendulum-v0")
# agent = SAC("mlp", env)
# trainer = OffPolicyTrainer(agent, env, log_interval=2, epochs=50, evaluate_episodes=10, max_ep_len=2000)

# trainer.train()
# trainer.evaluate()


def make_env(scenario_name, benchmark=False):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            scenario.isFinished,
        )
    else:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            None,
            scenario.isFinished,
        )
    return env


env = make_env("simple_spread")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_agents = env.n
agents = A2C(env)

agents.steps_per_episode = 300

trainer = OnPolicyTrainer(
    agent=agents,
    env=env,
    save_interval=500,
    steps_per_epoch=max_steps,
    epochs=max_episodes,
    log_interval=1,
    logdir="logs/",
)
trainer.train()
