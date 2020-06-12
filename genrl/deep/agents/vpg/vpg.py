from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.optim as opt
from torch.autograd import Variable

from ...common import (
    RolloutBuffer,
    get_env_properties,
    get_model,
    load_params,
    save_params,
    set_seeds,
    venv,
)


class VPG:
    """
    Vanilla Policy Gradient algorithm

    Paper https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf

    :param network_type: The deep neural network layer types ['mlp']
    :param env: The environment to learn from
    :param timesteps_per_actorbatch: timesteps per actor per update
    :param gamma: discount factor
    :param actor_batchsize: trajectories per optimizer epoch
    :param epochs: the optimizer's number of epochs
    :param lr_policy: policy network learning rate
    :param lr_value: value network learning rate
    :param save_interval: Number of episodes between saves of models
    :param tensorboard_log: the log location for tensorboard
    :param seed: seed for torch and gym
    :param device: device to use for tensor operations; 'cpu' for cpu and 'cuda' for gpu
    :param run_num: if model has already been trained
    :param save_model: True if user wants to save
    :param load_model: model loading path
    :param rollout_size: Rollout Buffer Size
    :type network_type: str
    :type env: Gym environment
    :type timesteps_per_actorbatch: int
    :type gamma: float
    :type actor_batchsize: int
    :type epochs: int
    :type lr_policy: float
    :type lr_value: float
    :type save_interval: int
    :type tensorboard_log: str
    :type seed: int
    :type device: str
    :type run_num: bool
    :type save_model: bool
    :type load_model: string
    :type rollout_size: int
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, venv],
        timesteps_per_actorbatch: int = 1000,
        gamma: float = 0.99,
        actor_batch_size: int = 4,
        epochs: int = 1000,
        lr_policy: float = 0.01,
        lr_value: float = 0.0005,
        policy_copy_interval: int = 20,
        layers: Tuple = (32, 32),
        tensorboard_log: str = None,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
        rollout_size: int = 2048,
    ):
        self.network_type = network_type
        self.env = env
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.gamma = gamma
        self.actor_batch_size = actor_batch_size
        self.epochs = epochs
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.save_interval = save_interval
        self.layers = layers
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save = save_params
        self.load = load_params
        self.rollout_size = rollout_size

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # init writer if tensorboard
        self.writer = None
        if self.tensorboard_log is not None:  # pragma: no cover
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.create_model()

    def create_model(self) -> None:
        """
        Initialize the actor and critic networks
        """
        state_dim, action_dim, discrete, action_lim = get_env_properties(self.env)
        print(state_dim, action_dim, discrete)
        # Instantiate networks and optimizers
        self.actor = get_model("p", self.network_type)(
            state_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        # load paramaters if already trained
        if self.load_model is not None:
            self.load(self)
            self.actor.load_state_dict(self.checkpoint["policy_weights"])

            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.optimizer_policy = opt.Adam(self.actor.parameters(), lr=self.lr_policy)

        self.rollout = RolloutBuffer(
            self.rollout_size,
            self.env.observation_space,
            self.env.action_space,
            n_envs=self.env.n_envs,
        )

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action for the given state

        :param state: State for which action has to be sampled
        :param deterministic: Whether the action is deterministic or not
        :type state: int, float, ...
        :type deterministic: bool
        :returns: The action
        :rtype: int, float, ...
        """
        state = Variable(torch.as_tensor(state).float().to(self.device))

        # create distribution based on policy_fn output
        a, c = self.actor.get_action(state, deterministic=False)

        return a, c.log_prob(a), None

    def get_value_log_probs(self, state, action):
        a, c = self.actor.get_action(state, deterministic=False)
        return c.log_prob(action)

    def get_traj_loss(self, value, done) -> None:
        """
        Calculates the loss for the trajectory
        """
        self.rollout.compute_returns_and_advantage(value.detach().cpu().numpy(), done)

    def update_policy(self) -> None:

        for rollout in self.rollout.get(256):

            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            log_prob = self.get_value_log_probs(rollout.observations, actions)

            policy_loss = rollout.returns * log_prob

            policy_loss = -torch.sum(policy_loss)

            loss = policy_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_policy.step()

    def collect_rollouts(self, initial_state):

        state = initial_state

        for i in range(2048):

            action, old_log_probs, _ = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action.numpy())
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                done,
                torch.Tensor([0] * self.env.n_envs),
                old_log_probs.detach(),
            )

            state = next_state

            for i, di in enumerate(done):
                if di:
                    self.rewards.append(self.epoch_reward[i])
                    self.epoch_reward[i] = 0

        return torch.Tensor([0] * self.env.n_envs), done

    def learn(self) -> None:  # pragma: no cover
        # training loop
        state = self.env.reset()
        for epoch in range(self.epochs):
            self.epoch_reward = np.zeros(self.env.n_envs)

            self.rollout.reset()
            self.rewards = []

            values, done = self.collect_rollouts(state)

            self.get_traj_loss(values, done)

            self.update_policy()

            if epoch % 1 == 0:
                print("Episode: {}, reward: {}".format(epoch, np.mean(self.rewards)))
                self.rewards = []
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", self.epoch_reward, epoch)

            if self.save_model is not None:
                if epoch % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, epoch)
                    print("Saved current model")

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "gamma": self.gamma,
            "actor_batch_size": self.actor_batch_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "weights": self.ac.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    algo = VPG("mlp", env)
    algo.learn()
    algo.evaluate(algo)
