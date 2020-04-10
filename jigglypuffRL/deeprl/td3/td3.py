import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
import gym

from jigglypuffRL.common import (
    get_model,
    evaluate,
    save_params,
    load_params,
)


class TD3:
    """
    Twin Delayed DDPG
    """
    def __init__(
        self,
        network_type,
        env,
        gamma=0.99,
        replay_size=1000000,
        batch_size=100,
        lr_p=0.001,
        lr_q=0.001,
        polyak=0.995,
        epochs=100,
        start_steps=10000,
        steps_per_epoch=4000,
        noise_std=0.1,
        max_ep_len=1000,
        start_update=1000,
        update_interval=50,
        save_interval=5000,
        layers=(256, 256),
        tensorboard_log=None,
        seed=None,
        render=False,
        device="cpu",
        pretrained=False,
        save_name=None,
        save_version=None,
    ):

        self.network_type = network_type
        self.env = env
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.polyak = polyak
        self.epochs = epochs
        self.start_steps = start_steps
        self.steps_per_epoch = steps_per_epoch
        self.noise_std = noise_std
        self.max_ep_len = max_ep_len
        self.start_update = start_update
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.layers = layers
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.render = render
        self.evaluate = evaluate
        self.pretrained = pretrained
        self.save_name = save_name
        self.save_version = save_version
        self.save = save_params
        self.load = load_params
        self.checkpoint = self.__dict__

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

        self.create_model()

    def create_model(self):
        state_dim, action_dim, disc = self.get_env_properties()

        self.ac = get_model("ac", self.network_type)(
            state_dim, action_dim, self.layers, "Qsa", False, True
        ).to(self.device)

        self.ac.qf1 = self.ac.critic
        self.ac.qf2 = get_model("q", self.network_type)(
            state_dim, action_dim, self.layers, "Qsa")

        self.ac_target = deepcopy(self.ac)

    def get_env_properties(self):
        state_dim = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            disc = True
        elif isinstance(self.env.action_space, gym.spaces.Box):
            a_dim = self.env.action_space.shape[0]
            disc = False
        else:
            raise NotImplementedError

        return state_dim, action_dim, disc

    def get_hyperparams(self):
        hyperparams = {
        "network_type" : self.network_type,
        "timesteps_per_actorbatch" : self.timesteps_per_actorbatch,
        "gamma" : self.gamma,
        "clip_param" : self.clip_param,
        "actor_batch_size" : self.actor_batch_size,
        "lr_policy" : self.lr_policy,
        "lr_value" : self.lr_value
        }

        return hyperparams