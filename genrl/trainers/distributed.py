from genrl.trainers import Trainer
import numpy as np
from genrl.utils import safe_mean


class DistributedTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.env = self.agent.env
        self._completed_training_flag = False

    def train(self, parameter_server_rref, experience_server_rref):
        raise NotImplementedError

    def train_wrapper(self, parameter_server_rref, experience_server_rref):
        self._completed_training_flag = False
        self.train(parameter_server_rref, experience_server_rref)
        self._completed_training_flag = True

    def is_done(self):
        return self._completed_training_flag

    def evaluate(self, timestep, render: bool = False) -> None:
        """Evaluate performance of Agent

        Args:
            render (bool): Option to render the environment during evaluation
        """
        episode_reward = 0
        episode_rewards = []
        state = self.env.reset()
        done = False
        for i in range(10):
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
