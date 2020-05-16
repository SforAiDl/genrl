import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
import gym
from copy import deepcopy

from genrl.deep.common import (
    get_model,
    evaluate,
    save_params,
    load_params,
    set_seeds,
)


class A2C:
    """
    Advantage Actor Critic algorithm (A2C)
    The synchronous version of A3C
    Paper: https://arxiv.org/abs/1602.01783

    :param network_type: The deep neural network layer types ['mlp']
    :param env: The environment to learn from
    :param gamma: Discount factor
    :param batch_size: Update batch size
    :param lr_actor: Policy Network learning rate
    :param lr_critic: Value Network learning rate
    :param num_episodes: Number of episodes
    :param steps_per_epoch: Number of timesteps per epoch
    :param max_ep_len: Maximum timesteps in an episode
    :param layers: Number of neurons in hidden layers
    :param tensorboard_log: The log location for Tensorboard(if None, no logging)
    :param seed: Seed for reproducing results
    :param render: True if environment is to be rendered, else False
    :param device: Device to use for Tensor operation ['cpu', 'cuda']
    :param run_num: Model run number if it has already been trained
    :param save_model: Directory the user wants to save models to
    :param save_interval: Number of steps between saves of models
    :type network_type: string
    :type env: Gym Environment
    :type gamma: float
    :type batch_size: int
    :type lr_a: float
    :type lr_c: float
    :type num_episodes: int
    :type steps_per_epoch: int
    :type max_ep_len: int
    :type layers: tuple or list
    :type tensorboard_log: string
    :type seed: int
    :type render: boolean
    :type device: string
    :type run_num: int
    :type save_model: string
    :type save_interval: int
    """
    def __init__(
        self,
        network_type,
        env,
        gamma=0.99,
        batch_size=64,
        lr_actor=0.01,
        lr_critic=0.1,
        num_episodes=400,
        steps_per_epoch=4000, 
        max_ep_len=1000,
        layers=(32, 32),
        tensorboard_log=None,
        seed=None,
        render=False,
        device='cpu',
        run_num=None,
        save_interval=5000,
    ):
        self.network_type = network_type
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.layers = layers
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.run_num = run_num
        self.save_interval = save_interval

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            set_seeds(seed, self.env)

        # Setup tensorboard writer 
        self.writer = None
        if self.tensorboard_log is not None:  # pragma: no cover
            from torch.utils.tensorboard import SummaryWriter
            
            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.create_model()

    def create_model(self):
        (
            state_dim,
            action_dim,
            discrete,
            action_lim
        ) = self.get_env_properties(self.env)

        self.ac = get_model("ac", self.network_type)(
            state_dim,
            action_dim,
            self.layers,
            "V",
            discrete,
            action_lim=action_lim
        ).to(self.device)

        self.actor_optimizer = opt.Adam(
            self.ac.actor.parameters(), lr=self.lr_actor
        )

        self.critic_optimizer = opt.Adam(
            self.ac.critic.parameters(), lr=self.lr_critic
        )

        self.traj_reward = []
        self.actor_hist = torch.Tensor().to(self.device)
        self.critic_hist = torch.Tensor().to(self.device)

        self.actor_loss_hist = torch.Tensor().to(self.device)
        self.critic_loss_hist = torch.Tensor().to(self.device)

        # load paramaters if already trained
        if self.run_num is not None:
            self.load(self)
            self.ac.actor.load_state_dict(self.checkpoint["actor_weights"])
            self.ac.critic.load_state_dict(self.checkpoint["critic_weights"])
            for key, item in self.checkpoint.items():
                if key not in ["actor_weights", "critic_weights"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

    def select_action(self, state, deterministic=True):
        state = torch.as_tensor(state).float().to(self.device)

        action, distribution = self.ac.get_action(state)
        log_prob = distribution.log_prob(action)
        value = self.ac.get_value(state)

        self.actor_hist = torch.cat(
            [self.actor_hist, log_prob.unsqueeze(0)]
        )
        self.critic_hist = torch.cat(
            [self.critic_hist, value.unsqueeze(0)]
        )

        action = action.detach().cpu().numpy()
        return action

    def get_traj_loss(self):
        discounted_reward = 0
        returns = []

        for reward in self.traj_reward[::-1]:
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = Variable(returns) - Variable(self.critic_hist)

        policy_loss = torch.mean(torch.mul(
            advantages,
            self.actor_hist.mul(-1)
        ))

        value_loss = nn.MSELoss()(
            self.critic_hist, Variable(returns)
        )

        self.actor_loss_hist = torch.cat([
            self.actor_loss_hist, policy_loss.unsqueeze(0)
        ])
        self.critic_loss_hist = torch.cat([
            self.critic_loss_hist, value_loss.unsqueeze(0)
        ])

        self.traj_reward = []
        self.actor_hist = torch.Tensor().to(self.device)
        self.critic_hist = torch.Tensor().to(self.device)

    def update(self, episode):
        policy_loss = torch.mean(self.actor_loss_hist)
        value_loss = torch.mean(self.critic_loss_hist)

        if self.tensorboard_log:
            self.writer.add_scalar("loss/actor", self.actor_loss, episode)
            self.writer.add_scalar("loss/critic", self.actor_loss, episode)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_loss_hist = torch.Tensor().to(self.device)
        self.critic_loss_hist = torch.Tensor().to(self.device)

    def learn(self):  # pragma: no cover
        for episode in range(self.num_episodes):
            episode_reward = 0
            steps = []
            for i in range(self.batch_size):
                state = self.env.reset()
                done = False

                for t in range(self.steps_per_epoch):
                    action = self.select_action(state)
                    state, reward, done, _ = self.env.step(action)

                    if self.render:
                        self.env.render()
                    
                    self.traj_reward.append(reward)

                    if done:
                        steps.append(t)
                        break
                
                episode_reward += np.sum(self.traj_reward) / self.batch_size
                self.get_traj_loss()
        
            self.update(episode)

            if episode % 5 == 0:
                print("Episode: {}, Reward: {}".format(
                    episode, episode_reward
                ))
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", episode_reward, episode)

            if self.run_num is not None:
                if episode % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, episode)
                    print("Saved current model")
        
        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_env_properties(self, env):
        state_dim = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            disc = True
            action_lim = None
        elif isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            action_lim = self.env.action_space.high[0]
            disc = False
        else:
            raise NotImplementedError

        return state_dim, action_dim, disc, action_lim

    def get_hyperparams(self):
        hyperparams = {
            "network_type": self.network_type,
            "steps_per_epoch": self.steps_per_epoch,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "actor_weights": self.ac.actor.state_dict(),
            "critic_weights": self.ac.critic.state_dict(),
        }

        return hyperparams



if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
    algo = A2C("mlp", env, device="cuda")
    algo.learn()
