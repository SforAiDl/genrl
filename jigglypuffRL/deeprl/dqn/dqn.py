import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

from jigglypuffRL.common import ReplayBuffer
from jigglypuffRL.deeprl.dqn.utils import DQN_Mlp


class DQN:
    """
    Deep Q Networks 
    Paper: https://arxiv.org/pdf/1312.5602.pdf
    :param network_type: (str) The deep neural network layer types ['Mlp']
    :param env: (Gym environment) The environment to learn from
    :param epochs: (int) Number of epochs
    :param max_iterations_per_epoch: (int) Number of iterations per epoch
    :param max_ep_len: (int) Maximum steps per episode
    :param gamma: (float) discount factor
    :param lr: (float) learing rate for the optimizer 
    :param batch_size: (int) Update batch size
    :param replay_size: (int) Replay memory size
    :param plot_loss_reward_graph: (bool) To print the loss and reward as a graph or text
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param seed : (int) seed for torch and gym
    :param render : (boolean) if environment is to be rendered
    :param device : (str) device to use for tensor operations; 'cpu' for cpu and 'cuda' for gpu
    """

    def __init__(
        self,
        network_type,
        env,
        epochs=100,
        max_iterations_per_epoch=100,
        max_ep_len=1000,
        gamma=0.99,
        lr=0.001,
        batch_size=32,
        replay_size=100,
        plot_loss_reward_graph=False,
        tensorboard_log=None,
        seed=None,
        render=False,
        device="cpu",
    ):
        self.env = env
        self.max_epochs = epochs
        self.max_iterations_per_epoch = max_iterations_per_epoch
        self.max_ep_len = max_ep_len
        self.replay_size = replay_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.plot_loss_reward_graph = plot_loss_reward_graph
        self.tensorboard_log = tensorboard_log
        self.render = render
        self.loss_hist = []
        self.reward_hist = []
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 500

        # Assign device
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Assign seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            self.env.seed(seed)
            random.seed(seed)

        # Setup tensorboard writer
        self.writer = None
        if self.tensorboard_log is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.tensorboard_log)

        self.create_model(network_type)

    def create_model(self, network_type):
        # self.dqn_network = get_dqn_network(network_type, self.env)
        if network_type == "MLP":
            self.model = DQN_Mlp(self.env)

        self.replay_buffer = ReplayBuffer(self.replay_size)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, state, frame_idx):
        epsilon = self.calculate_epsilon_by_frame(frame_idx)
        action = self.model.act(state, epsilon)
        return action

    def get_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action.long()))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        # print(f"q_values: {q_values}, next_q_values: {next_q_values}")
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        # print(f"q_value: {q_value}")
        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        self.loss_hist.append(loss)

        return loss

    def update_params(self):
        loss = self.get_td_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_epsilon_by_frame(self, frame_idx):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -1.0 * frame_idx / self.epsilon_decay
        )

    def plot(self, frame_idx):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title("frame %s. reward: %s" % (frame_idx, np.mean(self.reward_hist[-10:])))
        plt.plot(self.reward_hist)
        plt.subplot(132)
        plt.title("loss")
        plt.plot(self.loss_hist)
        plt.show()

    def learn(self):
        total_steps = self.max_epochs * self.max_iterations_per_epoch
        state, ep_r, ep, ep_len = self.env.reset(), 0, 0, 0

        for frame_idx in range(1, total_steps + 1):
            action = self.select_action(state, frame_idx)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push((state, action, reward, next_state, done))

            state = next_state
            ep_r += reward
            ep_len += 1

            done = False if ep_len == self.max_ep_len else done

            if done or (ep_len == self.max_ep_len):
                if ep % 20 == 0 and (self.plot_loss_reward_graph == False):
                    print(
                        "Ep: {}, reward: {}, frame_idx: {}".format(ep, ep_r, frame_idx)
                    )
                if self.tensorboard_log:
                    self.writer.add_scalar("episode_reward", ep_r, frame_idx)

                self.reward_hist.append(ep_r)
                state, ep_r, ep_len = self.env.reset(), 0, 0
                ep += 1

            if self.replay_buffer.get_len() > self.batch_size:
                self.update_params()

            if frame_idx % 200 == 0 and (self.plot_loss_reward_graph == True):
                self.plot(frame_idx)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    algo = DQN("MLP", env, plot_loss_reward_graph=True)
    algo.learn()
