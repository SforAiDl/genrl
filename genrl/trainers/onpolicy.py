import numpy as np

from genrl.trainers import Trainer


class OnPolicyTrainer(Trainer):
    """On Policy Trainer Class

    Trainer class for all the On Policy Agents: A2C, PPO1 and VPG

    Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        log_mode (:obj:`list` of str): List of different kinds of logging. Supported: ["csv", "stdout", "tensorboard"]
        log_key (str): Key plotted on x_axis. Supported: ["timestep", "episode"]
        log_interval (int): Timesteps between successive logging of parameters onto the console
        logdir (str): Directory where log files should be saved.
        epochs (int): Total number of epochs to train for
        max_timesteps (int): Maximum limit of timesteps to train for
        off_policy (bool): True if the agent is an off policy agent, False if it is on policy
        save_interval (int): Timesteps between successive saves of the agent's important hyperparameters
        save_model (str): Directory where the checkpoints of agent parameters should be saved
        run_num (int): A run number allotted to the save of parameters
        load_model (str): File to load saved parameter checkpoint from
        render (bool): True if environment is to be rendered during training, else False
        evaluate_episodes (int): Number of episodes to evaluate for
        seed (int): Set seed for reproducibility
    """

    def __init__(self, *args, **kwargs):
        super(OnPolicyTrainer, self).__init__(*args, **kwargs)

    def train(self) -> None:
        """Main training method"""
        if self.load_weights is not None or self.load_hyperparams is not None:
            self.load()

        for epoch in range(self.epochs):
            self.agent.epoch_reward = np.zeros(self.env.n_envs)

            self.agent.rollout.reset()

            state = self.env.reset()
            values, done = self.agent.collect_rollouts(state)

            self.agent.get_traj_loss(values, done)

            self.agent.update_params()

            if epoch % self.log_interval == 0:
                self.logger.write(
                    {
                        "timestep": epoch * self.agent.rollout_size,
                        "Epoch": epoch,  # This is not the same as an episode. 1 epoch is 1 rollout.
                        **self.agent.get_logging_params(),
                    },
                    self.log_key,
                )

            if (
                self.max_timesteps is not None
                and epoch * self.agent.rollout_size >= self.max_timesteps
            ):
                break

            if self.render:
                self.env.render()

            if self.save_interval != 0 and epoch % self.save_interval == 0:
                self.save(epoch * self.agent.batch_size)

        self.env.close()
        self.logger.close()
