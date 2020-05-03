import gym
import numpy as np

import torch
import torch.optim as opt
from torch.autograd import Variable
from copy import deepcopy

from genrl.deep.common import (
    ReplayBuffer,
    PrioritizedBuffer,
    get_model,
    evaluate,
    save_params,
    load_params,
    set_seeds,
)
from genrl.deep.agents.dqn.utils import (
    DuelingDQNValueMlp,
    NoisyDQNValue,
    CategoricalDQNValue,
)


class DQN:
    """
    Deep Q Networks
    Paper: (DQN) https://arxiv.org/pdf/1312.5602.pdf
    Paper: (Double DQN) https://arxiv.org/abs/1509.06461
    :param network_type: (str) The deep neural network layer types ['MLP']
    :param env: (Gym environment) The environment to learn from
    :param double_dqn: (boolean) For training Double DQN
    :param dueling_dqn: (boolean) For training Dueling DQN
    :param noisy_dqn: (boolean) For using Noisy Q
    :param categorical_dqn: (boolean) For using Distributional DQN
    :param parameterized_replay: (boolean) For using a prioritized buffer
    :param epochs: (int) Number of epochs
    :param max_iterations_per_epoch: (int) Number of iterations per epoch
    :param max_ep_len: (int) Maximum steps per episode
    :param gamma: (float) discount factor
    :param lr: (float) learing rate for the optimizer
    :param batch_size: (int) Update batch size
    :param replay_size: (int) Replay memory size
    :param tensorboard_log: (str) the log location for tensorboard
        (if None, no logging)
    :param seed : (int) seed for torch and gym
    :param render : (boolean) if environment is to be rendered
    :param device : (str) device to use for tensor operations; 'cpu' for cpu
        and 'cuda' for gpu
    """

    def __init__(
        self,
        network_type,
        env,
        double_dqn=False,
        dueling_dqn=False,
        noisy_dqn=False,
        categorical_dqn=False,
        prioritized_replay=False,
        epochs=100,
        max_iterations_per_epoch=100,
        max_ep_len=1000,
        gamma=0.99,
        lr=0.001,
        batch_size=32,
        replay_size=100,
        prioritized_replay_alpha=0.6,
        max_epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=1000,
        num_atoms=51,
        Vmin=-10,
        Vmax=10,
        tensorboard_log=None,
        seed=None,
        render=False,
        device="cpu",
        save_interval=5000,
        pretrained=None,
        run_num=None,
        save_model=None,
    ):
        self.env = env
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.noisy_dqn = noisy_dqn
        self.categorical_dqn = categorical_dqn
        self.prioritized_replay = prioritized_replay
        self.max_epochs = epochs
        self.max_iterations_per_epoch = max_iterations_per_epoch
        self.max_ep_len = max_ep_len
        self.replay_size = replay_size
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.tensorboard_log = tensorboard_log
        self.render = render
        self.loss_hist = []
        self.reward_hist = []
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.evaluate = evaluate
        self.run_num = run_num
        self.save_model = save_model
        self.save_interval = save_interval
        self.save = save_params
        self.load = load_params
        self.pretrained = pretrained

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

        self.create_model(network_type)

    def create_model(self, network_type):
        if network_type == "mlp":
            if self.dueling_dqn:
                self.model = DuelingDQNValueMlp(
                    self.env.observation_space.shape[0], self.env.action_space.n
                )

            elif self.categorical_dqn:
                self.model = CategoricalDQNValue(
                    self.env.observation_space.shape[0],
                    self.env.action_space.n,
                    self.num_atoms,
                    self.Vmin,
                    self.Vmax,
                )

            elif self.noisy_dqn:
                self.model = NoisyDQNValue(
                    self.env.observation_space.shape[0], self.env.action_space.n
                )
            else:
                self.model = get_model("v", network_type)(
                    self.env.observation_space.shape[0], self.env.action_space.n, "Qs"
                )
            # load paramaters if already trained
            if self.pretrained is not None:
                self.load(self)
                self.model.load_state_dict(self.checkpoint["weights"])
                for key, item in self.checkpoint.items():
                    if key not in ["weights", "save_model"]:
                        setattr(self, key, item)
                print("Loaded pretrained model")

            self.target_model = deepcopy(self.model)

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedBuffer(
                self.replay_size, self.prioritized_replay_alpha
            )

        else:
            self.replay_buffer = ReplayBuffer(self.replay_size)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        if np.random.rand() > self.epsilon:
            if self.categorical_dqn:
                state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
                dist = self.model(state).data.cpu()
                dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
                action = dist.sum(2).max(1)[1].numpy()[0]
            else:
                state = Variable(torch.FloatTensor(state))
                q_value = self.model(state)
                action = np.argmax(q_value.detach().numpy())
        else:
            action = self.env.action_space.sample()

        return action

    def get_td_loss(self):
        if self.prioritized_replay:
            (
                state,
                action,
                reward,
                next_state,
                done,
                indices,
                weight,
            ) = self.replay_buffer.sample(self.batch_size)
            weights = Variable(torch.FloatTensor(weight))
        else:
            state, action, reward, next_state, done = self.replay_buffer.sample(
                self.batch_size
            )

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action.long()))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        if self.categorical_dqn:
            proj_dist = self.projection_distribution(next_state, reward, done)
            dist = self.model(state)
            action = (
                action.unsqueeze(1)
                .unsqueeze(1)
                .expand(self.batch_size, 1, self.num_atoms)
            )
            dist = dist.gather(1, action).squeeze(1)
            dist.data.clamp_(0.01, 0.99)
            loss = -(Variable(proj_dist) * dist.log()).sum(1).mean()

        elif self.double_dqn:
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            q_next_state_values = self.model(next_state)
            action_next = q_next_state_values.max(1)[1]

            q_target_next_state_values = self.target_model(next_state)
            q_target_s_a_prime = q_target_next_state_values.gather(
                1, action_next.unsqueeze(1)
            ).squeeze(1)
            expected_q_value = reward + self.gamma * q_target_s_a_prime * (1 - done)

        else:
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            q_next_state_values = self.target_model(next_state)
            q_s_a_prime = q_next_state_values.max(1)[0]
            expected_q_value = reward + self.gamma * q_s_a_prime * (1 - done)

        if self.prioritized_replay and (not self.categorical_dqn):
            loss = (q_value - expected_q_value.detach()).pow(2) * weights
            priorities = loss + 1e-5
            loss = loss.mean()
            self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())

        elif (not self.prioritized_replay) and (not self.categorical_dqn):
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
        # loss = F.smooth_l1_loss(q_value,expected_q_value)

        else:
            pass

        self.loss_hist.append(loss)

        return loss

    def update_params(self):
        loss = self.get_td_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.noisy_dqn or self.categorical_dqn:
            self.model.reset_noise()
            self.target_model.reset_noise()

    def calculate_epsilon_by_frame(self, frame_idx):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
            -1.0 * frame_idx / self.epsilon_decay
        )

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = (
            next_action.unsqueeze(1)
            .unsqueeze(1)
            .expand(next_dist.size(0), 1, next_dist.size(2))
        )
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
            .long()
            .unsqueeze(1)
            .expand(self.batch_size, self.num_atoms)
        )

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        return proj_dist

    def learn(self):
        total_steps = self.max_epochs * self.max_iterations_per_epoch
        state, episode_reward, episode, episode_len = self.env.reset(), 0, 0, 0

        if self.double_dqn:
            self.update_target_model()

        for frame_idx in range(1, total_steps + 1):
            self.epsilon = self.calculate_epsilon_by_frame(frame_idx)
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if self.render:
                self.env.render()

            self.replay_buffer.push((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward
            episode_len += 1

            done = False if episode_len == self.max_ep_len else done

            if done or (episode_len == self.max_ep_len):
                if episode % 20 == 0:
                    print(
                        "Ep: {}, reward: {}, frame_idx: {}".format(
                            episode, episode_reward, frame_idx
                        )
                    )
                if self.tensorboard_log:
                    self.writer.add_scalar("episode_reward", episode_reward, frame_idx)

                self.reward_hist.append(episode_reward)
                state, episode_reward, episode_len = self.env.reset(), 0, 0
                episode += 1

            if self.replay_buffer.get_len() > self.batch_size:
                self.update_params()

            if self.save_model is not None:
                if frame_idx % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, frame_idx)
                    print("Saved current model")

            if frame_idx % 100 == 0:
                self.update_target_model()

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self):
        hyperparams = {
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "replay_size": self.replay_size,
            "double_dqn": self.double_dqn,
            "dueling_dqn": self.dueling_dqn,
            "noisy_dqn": self.noisy_dqn,
            "categorical_dqn": self.categorical_dqn,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "weights": self.model.state_dict(),
        }

        return hyperparams


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    algo = DQN("mlp", env, save_model="checkpoints")
    algo.learn()
