import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import gym

from ...common import (
    get_model,
    save_params,
    load_params,
    get_env_properties,
    set_seeds,
    RolloutBuffer,
    venv,
)

from typing import Union, Any, Optional, Tuple, Dict


class PPO1:
    """
    Proximal Policy Optimization algorithm (Clipped policy).

    Paper: https://arxiv.org/abs/1707.06347
    
    :param network_type: (str) The deep neural network layer types ['mlp']
    :param env: (Gym environment) The environment to learn from
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param gamma: (float) discount factor
    :param clip_param: (float) clipping parameter epsilon
    :param actor_batchsize: (int) trajectories per optimizer epoch
    :param epochs: (int) the optimizer's number of epochs
    :param lr_policy: (float) policy network learning rate
    :param lr_value: (float) value network learning rate
    :param policy_copy_interval: (int) number of optimizer before copying
        params from new policy to old policy
    :param save_interval: (int) Number of episodes between saves of models
    :param tensorboard_log: (str) the log location for tensorboard (if None,
        no logging)
    :param seed (int): seed for torch and gym
    :param device (str): device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :param run_num: (boolean) if model has already been trained
    :param save_model: (string) directory the user wants to save models to
    :param load_model: model loading path
    :type load_model: string
    """

    def __init__(
        self,
        network_type: str,
        env: Union[gym.Env, venv],
        timesteps_per_actorbatch: int = 256,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        actor_batch_size: int = 64,
        epochs: int = 1000,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        layers: Tuple = (64, 64),
        policy_copy_interval: int = 20,
        tensorboard_log: str = None,
        seed: Optional[int] = None,
        render: bool = False,
        device: Union[torch.device, str] = "cpu",
        run_num: int = None,
        save_model: str = None,
        load_model: str = None,
        save_interval: int = 50,
    ):
        self.network_type = network_type
        self.env = env
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.gamma = gamma
        self.clip_param = clip_param
        self.actor_batch_size = actor_batch_size
        self.epochs = epochs
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.layers = layers
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.policy_copy_interval = policy_copy_interval
        self.save_interval = save_interval
        self.run_num = run_num
        self.save_model = save_model
        self.load_model = load_model
        self.save = save_params
        self.load = load_params

        self.ent_coef = 0.01
        self.vf_coef = 0.5

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
        # Instantiate networks and optimizers
        state_dim, action_dim, disc, action_lim = self.get_env_properties(self.env)
        self.policy_new = get_model("p", self.network_type)(
            state_dim, action_dim, self.layers, disc=disc, action_lim=action_lim
        )
        self.policy_new = self.policy_new.to(self.device)

        self.value_fn = get_model("v", self.network_type)(state_dim, action_dim).to(
            self.device
        )

        # load paramaters if already trained
        if self.load_model is not None:
            self.load(self)
            self.policy_new.load_state_dict(self.checkpoint["policy_weights"])
            self.value_fn.load_state_dict(self.checkpoint["value_weights"])
            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.optimizer_policy = opt.Adam(
            self.policy_new.parameters(), lr=self.lr_policy
        )
        self.optimizer_value = opt.Adam(self.value_fn.parameters(), lr=self.lr_value)

        self.rollout = RolloutBuffer(
            2048,
            self.env.observation_space,
            self.env.action_space,
            n_envs=self.env.n_envs,
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(state).float().to(self.device)
        # create distribution based on policy output
        action, c_new = self.policy_new.get_action(state, deterministic=False)
        val = self.value_fn.get_value(state)

        return action.detach().cpu().numpy(), val, c_new.log_prob(action)

    def evaluate_actions(self, obs, old_actions):
        val = self.value_fn.get_value(obs)
        _, dist = self.policy_new.get_action(obs)
        return val, dist.log_prob(old_actions), dist.entropy()

    # get clipped loss for single trajectory (episode)
    def get_traj_loss(self, values, dones):
        self.rollout.compute_returns_and_advantage(values, dones, use_gae=True)

    def update_policy(self):

        for rollout in self.rollout.get(256):
            actions = rollout.actions

            if isinstance(self.env.action_space, gym.spaces.Discrete):
                actions = actions.long().flatten()

            # with torch.no_grad():
            values, log_prob, entropy = self.evaluate_actions(
                rollout.observations, actions
            )

            advantages = rollout.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(log_prob - rollout.old_log_prob)

            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            values = values.flatten()

            value_loss = F.mse_loss(rollout.returns, values)

            entropy_loss = -torch.mean(entropy)  # Change this to entropy

            loss = (
                policy_loss + self.ent_coef * entropy_loss
            )  # + self.vf_coef * value_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), 0.5)
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), 0.5)
            self.optimizer_value.step()

    def collect_rollouts(self, initial_state):

        state = initial_state

        for i in range(2048):

            with torch.no_grad():
                action, values, old_log_probs = self.select_action(state)

            next_state, reward, done, _ = self.env.step(np.array(action))
            self.epoch_reward += reward

            if self.render:
                self.env.render()

            self.rollout.add(
                state,
                action.reshape(self.env.n_envs, 1),
                reward,
                done,
                values,
                old_log_probs,
            )

            state = next_state

            for i, d in enumerate(done):
                if d:
                    self.rewards.append(self.epoch_reward[i])
                    self.epoch_reward[i] = 0

        return values, done

    def learn(self):  # pragma: no cover
        # training loop
        state = self.env.reset()
        for epoch in range(self.epochs):
            self.epoch_reward = np.zeros(self.env.n_envs)

            self.rollout.reset()
            self.rewards = []

            values, done = self.collect_rollouts(state)

            self.get_traj_loss(values.cpu().numpy(), done)

            self.update_policy()

            if epoch % 1 == 0:
                print("Episode: {}, reward: {}".format(epoch, np.mean(self.rewards)))
                self.rewards = []
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", self.epoch_reward, epoch)

            # if self.save_model is not None:
            #     if episode % self.save_interval == 0:
            #         self.checkpoint = self.get_hyperparams()
            #         self.save(self, episode)
            #         print("Saved current model")

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self) -> Dict[str, Any]:
        hyperparams = {
            "network_type": self.network_type,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "actor_batch_size": self.actor_batch_size,
            "lr_policy": self.lr_policy,
            "lr_value": self.lr_value,
            "policy_weights": self.policy_new.state_dict(),
            "value_weights": self.value_fn.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    algo = PPO1("mlp", env)
    algo.learn()
