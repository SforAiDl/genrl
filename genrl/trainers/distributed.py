from genrl.utils import safe_mean
from genrl.utils import Logger


class DistributedTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.env = self.agent.env
        self._completed_training_flag = False
        self.logger = Logger(formats=["stdout"])


    def train(self, parameter_server, experience_server):
        raise NotImplementedError

    def is_completed(self):
        return self._completed_training_flag

    def set_completed(self, value=True):
        self._completed_training_flag = value

    def evaluate(self, timestep, render: bool = False) -> None:
        """Evaluate performance of Agent

        Args:
            render (bool): Option to render the environment during evaluation
        """
        episode_rewards = []
        for i in range(5):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            episode_rewards.append(episode_reward)
            episode_reward = 0

        self.logger.write(
            {
                "timestep": timestep,
                **self.agent.get_logging_params(),
                "Episode Reward": safe_mean(episode_rewards),
            },
            "timestep",
        )
