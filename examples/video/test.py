from genrl.agents import DQN
from genrl.agents.deep.dqn.utils import ddqn_q_target, prioritized_q_loss
from genrl.environments import VectorEnv
from genrl.trainers import OffPolicyTrainer, OnPolicyTrainer

env = VectorEnv("CartPole-v0")
agent = DQN("mlp", env)
trainer = OffPolicyTrainer(
    agent, env, max_timesteps=15000, log_interval=25, evaluate_episodes=10
)
trainer.train()
trainer.evaluate()


# Double DQN only changes the get_target_q_values function of the Base DQN class
class DoubleDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, **kwargs)
        self._create_model()

    def get_target_q_values(self, next_states, rewards, dones):
        next_q_value_dist = self.model(next_states)
        next_best_actions = torch.argmax(next_q_value_dist, dim=-1).unsqueeze(-1)

        rewards, dones = rewards.unsqueeze(-1), dones.unsqueeze(-1)

        next_q_target_value_dist = self.target_model(next_states)
        max_next_q_target_values = next_q_target_value_dist.gather(2, next_best_actions)
        target_q_values = rewards + agent.gamma * torch.mul(
            max_next_q_target_values, (1 - dones)
        )
        return target_q_values


# Adapt a dueling network that uses a DDQN target
class DuelingDQN(DQN):
    def __init__(self, *args, buffer_type="push", **kwargs):
        super(DuelingDQN, self).__init__(*args, buffer_type=buffer_type, **kwargs)
        self.dqn_type = "dueling"  # Uses a Dueling Value Architecture
        self._create_model()

    def get_target_q_values(self, *args):
        return ddqn_q_target(self, *args)

    # Uncomment the next 2 lines if Prioritized Replay is needed
    # def get_q_loss(self, *args):
    #     return prioritized_q_loss(self, *args)


env = VectorEnv("CartPole-v0")
agent = DuelingDQN("mlp", env)
trainer = OffPolicyTrainer(
    agent, env, max_timesteps=15000, log_interval=25, evaluate_episodes=10
)
trainer.train()
trainer.evaluate()
