import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
import gym

from ...common import (
    get_model,
    evaluate,
    save_params,
    load_params,
    set_seeds,
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
    :param policy_copy_interval: number of optimizer before copying params from new policy to old policy
    :param save_interval: Number of episodes between saves of models
    :param tensorboard_log: the log location for tensorboard
    :param seed: seed for torch and gym
    :param device: device to use for tensor operations; 'cpu' for cpu and 'cuda' for gpu
    :param run_num: if model has already been trained
    :param save_model: True if user wants to save model
    :type network_type: str
    :type env: Gym environment
    :type timesteps_per_actorbatch: int
    :type gamma: float
    :type actor_batchsize: int
    :type epochs: int
    :type lr_policy: float
    :type lr_value: float
    :type policy_copy_interval: int
    :type save_interval: int
    :type tensorboard_log: str
    :type seed: int
    :type device: str
    :type run_num: bool
    :type save_model: bool
    """

    def __init__(
        self,
        network_type,
        env,
        timesteps_per_actorbatch=1000,
        gamma=0.99,
        actor_batch_size=4,
        epochs=1000,
        lr_policy=0.01,
        lr_value=0.0005,
        policy_copy_interval=20,
        pretrained=None,
        layers=(32, 32),
        tensorboard_log=None,
        seed=None,
        render=False,
        device="cpu",
        run_num=None,
        save_model=None,
        save_interval=50,
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
        self.policy_copy_interval = policy_copy_interval
        self.evaluate = evaluate
        self.save_interval = save_interval
        self.pretrained = pretrained
        self.layers = layers
        self.run_num = run_num
        self.save_model = save_model
        self.save = save_params
        self.load = load_params

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

    def create_model(self):
        '''
        Initialize the actor and critic networks 
        '''
        state_dim, action_dim, action_lim, discrete = self.get_env_properties()
        # Instantiate networks and optimizers
        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "V", discrete, action_lim=action_lim
        ).to(self.device)

        # load paramaters if already trained
        if self.pretrained is not None:
            self.load(self)
            self.ac.actor.load_state_dict(self.checkpoint["policy_weights"])
            self.ac.critic.load_state_dict(self.checkpoint["value_weights"])

            for key, item in self.checkpoint.items():
                if key not in ["policy_weights", "value_weights", "save_model"]:
                    setattr(self, key, item)
            print("Loaded pretrained model")

        self.optimizer_policy = opt.Adam(self.ac.actor.parameters(), lr=self.lr_policy)
        self.optimizer_value = opt.Adam(self.ac.critic.parameters(), lr=self.lr_value)

        self.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.value_hist = Variable(torch.Tensor()).to(self.device)
        self.traj_reward = []
        self.policy_loss_hist = Variable(torch.Tensor()).to(self.device)
        self.value_loss_hist = Variable(torch.Tensor()).to(self.device)

    def get_env_properties(self):
        '''
        Helper function to extract the observation and action space

        :returns: Observation space, Action Space and whether the action space is discrete or not 
        :rtype: int, float, ... ; int, float, ... ; bool
        '''
        state_dim = self.env.observation_space.shape[0]
        action_lim = None

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            disc = True
        elif isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            action_lim = self.env.action_space.high[0]
            disc = False
        else:
            raise NotImplementedError

        return state_dim, action_dim, action_lim, disc

    def select_action(self, state, deterministic=False):
        '''
        Select action for the given state 

        :param state: State for which action has to be sampled
        :param deterministic: Whether the action is deterministic or not 
        :type state: int, float, ...
        :type deterministic: bool
        :returns: The action 
        :rtype: int, float, ...
        '''
        state = Variable(torch.as_tensor(state).float().to(self.device))

        # create distribution based on policy_fn output
        a, c = self.ac.get_action(state, deterministic=False)
        val = self.ac.get_value(state).unsqueeze(0)

        # store policy probs and value function for current trajectory
        self.policy_hist = torch.cat([self.policy_hist, c.log_prob(a).unsqueeze(0)])

        self.value_hist = torch.cat([self.value_hist, val])

        return a

    def get_traj_loss(self):
        '''
        Calculates the loss for the trajectory 
        '''
        disc_R = 0
        returns = []

        # calculate discounted return
        for reward in self.traj_reward[::-1]:
            disc_R = reward + self.gamma * disc_R
            returns.insert(0, disc_R)

        # advantage estimation
        returns = torch.FloatTensor(returns).to(self.device)
        advantage = Variable(returns) - Variable(self.value_hist)

        # compute policy and value loss
        loss_policy = torch.sum(
            torch.mul(self.policy_hist, advantage).mul(-1), -1
        ).unsqueeze(0)

        loss_value = nn.MSELoss()(self.value_hist, Variable(returns)).unsqueeze(0)

        # store traj loss values in epoch loss tensors
        self.policy_loss_hist = torch.cat([self.policy_loss_hist, loss_policy])
        self.value_loss_hist = torch.cat([self.value_loss_hist, loss_value])

        # clear traj history
        self.traj_reward = []
        self.policy_hist = Variable(torch.Tensor()).to(self.device)
        self.value_hist = Variable(torch.Tensor()).to(self.device)

    def update_policy(self, episode, copy_policy=False):
        '''
        Update the policy and take the step for the optimizer

        :param episode: Episode number 
        :param copy_policy: Whether you want to copy the policy or not 
        :type episode: int 
        :type copy_policy: bool
        '''
        # mean of all traj losses in single epoch
        loss_policy = torch.mean(self.policy_loss_hist)
        loss_value = torch.mean(self.value_loss_hist)

        # tensorboard book-keeping
        if self.tensorboard_log:
            self.writer.add_scalar("loss/policy", loss_policy, episode)
            self.writer.add_scalar("loss/value", loss_value, episode)

        # take gradient step
        self.optimizer_policy.zero_grad()
        loss_policy.backward()  # B
        self.optimizer_policy.step()

        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()

        # clear loss history for epoch
        self.policy_loss_hist = Variable(torch.Tensor()).to(self.device)
        self.value_loss_hist = Variable(torch.Tensor()).to(self.device)

        if copy_policy:
            pass

    def learn(self):  # pragma: no cover
        # training loop
        for episode in range(self.epochs):
            epoch_reward = 0
            for i in range(self.actor_batch_size):
                state = self.env.reset()
                done = False
                for t in range(self.timesteps_per_actorbatch):
                    action = Variable(self.select_action(state, deterministic=False))
                    state, reward, done, _ = self.env.step(action.item())

                    if self.render:
                        self.env.render()

                    self.traj_reward.append(reward)

                    if done:
                        break

                epoch_reward += np.sum(self.traj_reward) / self.actor_batch_size
                self.get_traj_loss()

            self.update_policy(episode)

            if episode % 20 == 0:
                print("Episode: {}, reward: {}".format(episode, epoch_reward))
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", epoch_reward, episode)

            if self.save_model is not None:
                if episode % self.save_interval == 0:
                    self.checkpoint = self.get_hyperparams()
                    self.save(self, episode)
                    print("Saved current model")

        self.env.close()
        if self.tensorboard_log:
            self.writer.close()

    def get_hyperparams(self):
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
