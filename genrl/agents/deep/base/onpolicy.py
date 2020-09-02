import torch

from genrl.agents.deep.base import BaseAgent
from genrl.core import RolloutBuffer


class OnPolicyAgent(BaseAgent):
    """Base On Policy Agent Class

    Attributes:
        network (str): The network type of the Q-value function.
            Supported types: ["cnn", "mlp"]
        env (Environment): The environment that the agent is supposed to act on
        create_model (bool): Whether the model of the algo should be created when initialised
        batch_size (int): Mini batch size for loading experiences
        gamma (float): The discount factor for rewards
        layers (:obj:`tuple` of :obj:`int`): Layers in the Neural Network
            of the Q-value function
        lr_policy (float): Learning rate for the policy/actor
        lr_value (float): Learning rate for the Q-value function
        rollout_size (int): Capacity of the Rollout Buffer
        buffer_type (str): Choose the type of Buffer: ["rollout"]
        seed (int): Seed for randomness
        render (bool): Should the env be rendered during training?
        device (str): Hardware being used for training. Options:
            ["cuda" -> GPU, "cpu" -> CPU]
    """

    def __init__(
        self, *args, rollout_size: int = 1024, buffer_type: str = "rollout", **kwargs
    ):
        super(OnPolicyAgent, self).__init__(*args, **kwargs)
        self.rollout_size = rollout_size

        gae_lambda = kwargs["gae_lambda"] if "gae_lambda" in kwargs else 1.0

        if buffer_type == "rollout":
            self.rollout = RolloutBuffer(
                self.rollout_size, self.env, gae_lambda=gae_lambda
            )
        else:
            raise NotImplementedError

    def update_params(self) -> None:
        """Update parameters of the model"""
        raise NotImplementedError

    def collect_rewards(self, dones: torch.Tensor, timestep: int):
        """Helper function to collect rewards

        Runs through all the envs and collects rewards accumulated during rollouts

        Args:
            dones (:obj:`torch.Tensor`): Game over statuses of each environment
            timestep (int): Timestep during rollout
        """
        for i, done in enumerate(dones):
            if done or timestep == self.rollout_size - 1:
                self.rewards.append(self.env.episode_reward[i].detach().clone())
                self.env.reset_single_env(i)

    def collect_rollouts(self, state: torch.Tensor):
        """Function to collect rollouts

        Collects rollouts by playing the env like a human agent and inputs information into
        the rollout buffer.

        Args:
            state (:obj:`torch.Tensor`): The starting state of the environment

        Returns:
            values (:obj:`torch.Tensor`): Values of states encountered during the rollout
            dones (:obj:`torch.Tensor`): Game over statuses of each environment
        """
        for i in range(self.rollout_size):
            action, values, old_log_probs = self.select_action(state)

            next_state, reward, dones, _ = self.env.step(action)

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                dones,
                values.detach(),
                old_log_probs.detach(),
            )

            state = next_state

            self.collect_rewards(dones, i)

        return values, dones
