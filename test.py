from genrl.deep.agents import DQN, DoubleDQN, DuelingDQN, NoisyDQN
from genrl.deep.agents.dqn.utils import ddqn_q_target
from genrl.deep.common.trainer import OffPolicyTrainer
from genrl.environments import VectorEnv


class CustomDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(CustomDQN, self).__init__(*args, **kwargs)

    def get_target_q_values(self, *args):
        return ddqn_q_target(self, *args)


env = VectorEnv("CartPole-v0")
agent = CustomDQN("mlp", env, replay_size=500, lr_value=0.01)
print(agent.model)
trainer = OffPolicyTrainer(agent, env, log_interval=10)
trainer.train()
trainer.evaluate()
