import torch

from genrl.trainers import OffPolicyTrainer


class OfflineTrainer(OffPolicyTrainer):
    """Offline RL Trainer Class

    Trainer class for all the Offline RL Agents: BCQ (more to be added)

    Attributes:
        agent (object): Agent algorithm object
        env (object): Environment
        buffer (object): Replay Buffer object
        buffer_path (str): Path to the saved buffer file
        max_ep_len (int): Maximum Episode length for training
        max_timesteps (int): Maximum limit of timesteps to train for
        warmup_steps (int): Number of warmup steps. (random actions are taken to add randomness to training)
        start_update (int): Timesteps after which the agent networks should start updating
        update_interval (int): Timesteps between target network updates
        log_mode (:obj:`list` of str): List of different kinds of logging. Supported: ["csv", "stdout", "tensorboard"]
        log_key (str): Key plotted on x_axis. Supported: ["timestep", "episode"]
        log_interval (int): Timesteps between successive logging of parameters onto the console
        logdir (str): Directory where log files should be saved.
        epochs (int): Total number of epochs to train for
        off_policy (bool): True if the agent is an off policy agent, False if it is on policy
        save_interval (int): Timesteps between successive saves of the agent's important hyperparameters
        save_model (str): Directory where the checkpoints of agent parameters should be saved
        run_num (int): A run number allotted to the save of parameters
        load_model (str): File to load saved parameter checkpoint from
        render (bool): True if environment is to be rendered during training, else False
        evaluate_episodes (int): Number of episodes to evaluate for
        seed (int): Set seed for reproducibility
    """

    def __init__(self, *args, buffer_path: str = None, **kwargs):
        super(OfflineTrainer, self).__init__(*args, **kwargs)
        self.buffer_path = buffer_path

        if self.buffer_path is None:
            self.generate_buffer("random")

    def generate_buffer(self, generate_type: str = "random") -> None:
        """Make a replay buffer from a specific kind of agent

        Args:
            generate_type (str): Type of generation for the buffer. Can choose from ["random", "agent"]
                Not generatable at the moment.
        """
        raise NotImplementedError

    def train(self) -> None:
        """Main training method"""

        self.buffer.load(self.buffer_path)

        state = self.env.reset()
        self.noise_reset()

        self.training_rewards = []
        self.episodes = 0

        for timestep in range(0, self.max_timesteps, self.env.n_envs):
            done = self.env.dones

            if self.check_game_over_status(timestep, done):
                self.noise_reset()

                if self.episodes % self.log_interval == 0:
                    self.log(timestep)

            if timestep >= self.start_update and timestep % self.update_interval == 0:
                self.agent.update_params(self.update_interval)

            if (
                timestep >= self.start_update
                and self.save_interval != 0
                and timestep % self.save_interval == 0
            ):
                self.save(timestep)

        self.env.close()
        self.logger.close()
