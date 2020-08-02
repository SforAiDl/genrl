import shutil

from genrl import A2C, DDPG, PPO1, SAC, TD3, VPG
from genrl.deep.common import (
    MlpActorCritic,
    MlpPolicy,
    MlpValue,
    NormalActionNoise,
    OffPolicyTrainer,
    OnPolicyTrainer,
    OrnsteinUhlenbeckActionNoise,
)
from genrl.environments import VectorEnv


class custom_policy(MlpPolicy):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_policy, self).__init__(state_dim, action_dim, kwargs.get("hidden"))
        self.state_dim = state_dim
        self.action_dim = action_dim


class custom_actorcritic(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_actorcritic, self).__init__(
            state_dim, action_dim, kwargs.get("hidden")
        )
        self.state_dim = state_dim
        self.action_dim = action_dim


class custom_multiactorcritic(MlpActorCritic):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(custom_multiactorcritic, self).__init__(
            state_dim, action_dim, kwargs.get("hidden"), val_type="Qsa", discrete=False
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qf2 = MlpValue(
            state_dim, action_dim, val_type="Qsa", hidden=kwargs.get("hidden")
        )
        self.qf1 = self.critic


def test_custom_vpg():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = custom_policy(state_dim, action_dim, hidden=(64, 64))

    algo = VPG(policy, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_ppo1_custom():
    env = VectorEnv("CartPole-v0", 1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actorcritic = custom_actorcritic(state_dim, action_dim, hidden=(64, 64))

    algo = PPO1(actorcritic, env)

    trainer = OnPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")


def test_td3_custom():
    env = VectorEnv("Pendulum-v0", 2)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    multiactorcritic = custom_multiactorcritic(state_dim, action_dim, hidden=(1, 1))

    algo = TD3(multiactorcritic, env, noise=OrnsteinUhlenbeckActionNoise)

    trainer = OffPolicyTrainer(algo, env, log_mode=["csv"], logdir="./logs", epochs=1)
    trainer.train()
    shutil.rmtree("./logs")
