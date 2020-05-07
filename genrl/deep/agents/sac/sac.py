import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import gym
from copy import deepcopy
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from genrl.deep.common import (
    get_model,
    ReplayBuffer,
    save_params,
    load_params,
    evaluate,
    set_seeds,
)


class SAC:
    """
    Soft Actor Critic algorithm (SAC)
    Paper: https://arxiv.org/abs/1812.05905
    :param network_type: (str) The deep neural network layer types ['mlp']
    :param env: (Gym environment) The environment to learn from
    :param gamma: (float) discount factor
    :param replay_size: (int) Replay memory size
    :param batch_size: (int) Update batch size
    :param lr: (float) network learning rate
    :param alpha: (float) entropy weight
    :param polyak: (float) Polyak averaging weight to update target network
    :param epochs: (int) Number of epochs
    :param start_steps: (int) Number of exploratory steps at start
    :param steps_per_epoch: (int) Number of steps per epoch
    :param max_ep_len: (int) Maximum steps per episode
    :param start_update: (int) Number of steps before first parameter update
    :param update_interval: (int) Number of steps between parameter updates
    :param save_interval: (int) Number of steps between saves of models
    :param layers: (tuple or list) Number of neurons in hidden layers
    :param tensorboard_log: (str) the log location for tensorboard (if None,
        no logging)
    :param seed (int): seed for torch and gym
    :param render (boolean): if environment is to be rendered
    :param device (str): device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    :param run_num: (boolean) if model has already been trained
    :param save_name: (str) model save name (if None, model hasn't been
        pretrained)
    :param save_version: (int) model save version (if None, model hasn't been
        pretrained)
    """

    def __init__(
        self,
        network_type,
        env,
        gamma=0.99,
        replay_size=1000000,
        batch_size=256,
        lr=3e-4,
        alpha=0.01,
        polyak=0.995,
        entropy_tuning=True,
        epochs=1000,
        start_steps=0,
        steps_per_epoch=1000,
        max_ep_len=1000,
        start_update=256,
        update_interval=1,
        layers=(256, 256),
        pretrained=None,
        tensorboard_log=None,
        seed=None,
        render=False,
        device="cpu",
        run_num=None,
        save_model=None,
        save_interval=5000,
    ):

        self.network_type = network_type
        self.env = env
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.polyak = polyak
        self.entropy_tuning = entropy_tuning
        self.epochs = epochs
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.start_update = start_update
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.layers = layers
        self.pretrained = pretrained
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.run_num = run_num
        self.save_model = save_model
        self.save = save_params
        self.load = load_params
        self.evaluate = evaluate

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # Setup tensorboard writer
        self.writer = None
        if self.tensorboard_log is not None: #pragma: no cover
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.create_model()

    def create_model(self):
        state_dim = self.env.observation_space.shape[0]

        # initialize models
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            disc = True
        elif isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            disc = False
        else:
            raise NotImplementedError

        self.q1 = get_model("v", self.network_type)(
            state_dim, action_dim, "Qsa", self.layers
        ).to(self.device)
        self.q2 = get_model("v", self.network_type)(
            state_dim, action_dim, "Qsa", self.layers
        ).to(self.device)
        self.policy = get_model("p", self.network_type)(
            state_dim, action_dim, self.layers, disc, False, sac=True
        )

        if self.pretrained is not None:
            self.load(self)
            self.q1.load_state_dict(self.checkpoint["q1_weights"])
            self.q2.load_state_dict(self.checkpoint["q2_weights"])
            self.policy.load_state_dict(self.checkpoint["policy_weights"])

            for key, item in self.checkpoint.items():
                if key not in ["weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.q1_targ = deepcopy(self.q1).to(self.device)
        self.q2_targ = deepcopy(self.q2).to(self.device)

        # freeze target parameters
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        # optimizers
        self.q1_optimizer = opt.Adam(self.q1.parameters(), self.lr)
        self.q2_optimizer = opt.Adam(self.q2.parameters(), self.lr)
        self.policy_optimizer = opt.Adam(self.policy.parameters(), self.lr)

        if self.entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=self.lr)

        self.replay_buffer = ReplayBuffer(self.replay_size)

        # set action scales
        if self.env.action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.0
            )
            self.action_bias = torch.FloatTensor(
                (self.env.action_space.high + self.env.action_space.low) / 2.0
            )

    def sample_action(self, state):
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()

        # reparameterization trick
        distribution = Normal(mean, std)
        xi = distribution.rsample()
        yi = torch.tanh(xi)
        action = yi * self.action_scale + self.action_bias
        log_pi = distribution.log_prob(xi)

        # enforcing action bound (appendix of paper)
        log_pi -= torch.log(
            self.action_scale * (1 - yi.pow(2)) + np.finfo(np.float32).eps
        )
        log_pi = log_pi.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_pi, mean

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.sample_action(state)
        return action.detach().cpu().numpy()[0]

    def update_params(self, state, action, reward, next_state, done):
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        # compute targets
        with torch.no_grad():
            next_action, next_log_pi, _ = self.sample_action(next_state)
            next_q1_targ = self.q1_targ(torch.cat([next_state, next_action], dim=-1))
            next_q2_targ = self.q2_targ(torch.cat([next_state, next_action], dim=-1))
            next_q_targ = (
                torch.min(next_q1_targ, next_q2_targ) - self.alpha * next_log_pi
            )
            # print(reward.shape, done.shape, next_q_targ.shape)
            next_q = reward + self.gamma * (1 - done.squeeze()) * next_q_targ

        # compute losses
        q1 = self.q1(torch.cat([state, action], dim=-1))
        q2 = self.q2(torch.cat([state, action], dim=-1))

        q1_loss = nn.MSELoss()(q1, next_q)
        # print(q1.shape, next_q.shape)
        q2_loss = nn.MSELoss()(q2, next_q)

        pi, log_pi, _ = self.sample_action(state)
        q1_pi = self.q1(torch.cat([state, pi], dim=-1))
        q2_pi = self.q2(torch.cat([state, pi], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # gradient step
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # alpha loss
        if self.entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)

        # soft update target params
        for target_param, param in zip(self.q1_targ.parameters(), self.q1.parameters()):
            target_param.data.copy_(
                target_param.data * self.polyak + param.data * (1 - self.polyak)
            )
        for target_param, param in zip(self.q2_targ.parameters(), self.q2.parameters()):
            target_param.data.copy_(
                target_param.data * self.polyak + param.data * (1 - self.polyak)
            )

        return (q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item())

    def learn(self): #pragma: no cover
        if self.tensorboard_log:
            writer = SummaryWriter(self.tensorboard_log)

        total_steps = self.steps_per_epoch * self.epochs * self.env.n_envs

        episode_reward, episode_len = np.zeros(self.env.n_envs), np.zeros(self.env.n_envs)
        state = self.env.reset()
        for i in range(0, total_steps, self.env.n_envs):
            # done = [False] * self.env.n_envs

            # while not done:
                # sample action
            if i > self.start_steps:
                action = self.select_action(state)
            else:
                action = self.env.sample()

            if (
                i >= self.start_update
                and i % self.update_interval == 0
                and self.replay_buffer.get_len() > self.batch_size
            ):
                # get losses
                batch = self.replay_buffer.sample(self.batch_size)
                states, actions, rewards, next_states, dones = (
                    x.to(self.device) for x in batch
                )
                q1_loss, q2_loss, policy_loss, alpha_loss = self.update_params(
                    states, actions, rewards, next_states, dones
                )

                # write loss logs to tensorboard
                if self.tensorboard_log:
                    writer.add_scalar("loss/q1_loss", q1_loss, i)
                    writer.add_scalar("loss/q2_loss", q2_loss, i)
                    writer.add_scalar("loss/policy_loss", policy_loss, i)
                    writer.add_scalar("loss/alpha_loss", alpha_loss, i)

                if self.save_model is not None:
                    if i >= self.start_update and i % self.save_interval == 0:
                        self.checkpoint = self.get_hyperparams()
                        self.save(self, i)
                        print("Saved current model")

                # prepare transition for replay memory push
            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
            
            done = [False if ep_len==self.max_ep_len else done for ep_len in episode_len]

            if np.any(done) or np.any(episode_len == self.max_ep_len):
                for i, d in enumerate(done):
                    if d:
                        episode_reward[i] = 0
                        episode_len[i] = 0

            self.replay_buffer.extend(zip(state, action, reward, next_state, done))
            state = next_state

            if i > total_steps:
                break

            # write episode reward to tensorboard logs
            if self.tensorboard_log:
                writer.add_scalar("reward/episode_reward", episode_reward, i)

            if sum(episode_len) % (5*self.env.n_envs) == 0 and sum(episode_len) != 0:
                print(
                    "Episode: {}, total numsteps: {}, reward: {}".format(
                        sum(episode_len), i, episode_reward
                    )
                )
            # ep += 1

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self):
        hyperparams = {
            "network_type": self.network_type,
            "gamma": self.gamma,
            "lr": self.lr,
            "replay_size": self.replay_size,
            "entropy_tuning": self.entropy_tuning,
            "alpha": self.alpha,
            "polyak": self.polyak,
            "q1_weights": self.q1.state_dict(),
            "q2_weights": self.q2.state_dict(),
            "policy_weights": self.policy.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    algo = SAC("mlp", env, seed=0)
    algo.learn()
    algo.evaluate(algo)
